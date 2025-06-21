"""
Hybrid RGB-Multispectral model for plant disease classification.

This model implements a dual-stream architecture that processes RGB and multispectral
(MS) data in parallel, with optional feature fusion. It's designed to work with both
real and synthetic MS data, with robust handling for missing modalities.
"""
from typing import Any, Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class HybridModel(BaseModel):
    """
    A hybrid model that fuses features from RGB and multispectral (MS) data streams.
    
    The model supports:
    - RGB-only mode (when use_ms=False)
    - RGB + Real MS data
    - RGB + Synthetic MS data (with fallback to RGB-only when MS is not available)
    - Multiple fusion strategies (concat, add, attention)
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-4,
        use_ms: bool = False,
        backbone_name: str = 'efficientnet_b0',
        ms_channels: int = 3,  # Default to 3 for compatibility with synthetic MS
        pretrained_rgb: bool = True,
        pretrained_ms: bool = False,
        fusion_method: str = 'concat',  # 'concat', 'add', or 'attention'
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize the HybridModel.
        
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate for the optimizer
            use_ms: Whether to use multispectral data
            backbone_name: Name of the backbone architecture (from timm)
            ms_channels: Number of channels in MS data
            pretrained_rgb: Use pretrained weights for RGB backbone
            pretrained_ms: Use pretrained weights for MS backbone (if different from RGB)
            fusion_method: How to fuse RGB and MS features ('concat', 'add', 'attention')
            dropout_rate: Dropout rate for the fusion layers
            **kwargs: Additional arguments passed to BaseModel
        """
        self.save_hyperparameters()
        
        # Store config parameters
        self.use_ms = use_ms
        self.backbone_name = backbone_name
        self.ms_channels = ms_channels
        self.pretrained_rgb = pretrained_rgb
        self.pretrained_ms = pretrained_ms
        self.fusion_method = fusion_method
        self.dropout_rate = dropout_rate

        # Initialize the model architecture
        super().__init__(num_classes=num_classes, learning_rate=learning_rate, **kwargs)

    def _build_model(self):
        """Build the model architecture."""
        # --- RGB Branch (always present) ---
        self.rgb_backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained_rgb,
            num_classes=0,  # Return features, not logits
            features_only=False
        )
        
        # Get feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            rgb_features = self.rgb_backbone(dummy_input)
            self.rgb_features_dim = rgb_features.shape[1]
        
        if not self.use_ms:
            # If not using MS data, use a simple classifier
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.rgb_features_dim, self.hparams.num_classes)
            )
            return
            
        # --- Multispectral Branch (optional) ---
        # Adapter to convert MS to 3 channels if needed
        if self.ms_channels != 3:
            self.ms_adapter = nn.Sequential(
                nn.Conv2d(self.ms_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1)
            )
        
        # MS backbone (can be same or different from RGB backbone)
        self.ms_backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained_ms,
            num_classes=0,
            features_only=False
        )
        
        # Initialize MS backbone with RGB weights if specified
        if self.pretrained_ms and not self.pretrained_rgb:
            self._init_ms_backbone()
        
        # --- Feature Fusion ---
        if self.fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(self.rgb_features_dim * 2, self.rgb_features_dim),
                nn.BatchNorm1d(self.rgb_features_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
            )
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            self.attention = nn.Sequential(
                nn.Linear(self.rgb_features_dim * 2, self.rgb_features_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.rgb_features_dim // 2, 2),
                nn.Softmax(dim=1)
            )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.rgb_features_dim, self.hparams.num_classes)
        )


    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass that handles both RGB and optional MS data.
        
        Args:
            batch: Dictionary containing:
                - 'image': RGB image tensor [B, 3, H, W]
                - 'ms_data': (Optional) MS image tensor [B, C, H, W]
                
        Returns:
            Classification logits [B, num_classes]
        """
        # RGB features (always present)
        x_rgb = batch['image']
        rgb_features = self.rgb_backbone(x_rgb)
        
        # Early return if not using MS data or MS data is not available
        if not self.use_ms or 'ms_data' not in batch or batch['ms_data'] is None:
            return self.classifier(rgb_features)
            
        # Process MS data
        x_ms = batch['ms_data']
        
        # Adapt MS channels if needed
        if hasattr(self, 'ms_adapter'):
            x_ms = self.ms_adapter(x_ms)
            
        # Get MS features
        ms_features = self.ms_backbone(x_ms)
        
        # Feature fusion
        if self.fusion_method == 'concat':
            # Simple concatenation
            fused_features = torch.cat([rgb_features, ms_features], dim=1)
            fused_features = self.fusion(fused_features)
        elif self.fusion_method == 'add':
            # Element-wise addition
            fused_features = rgb_features + ms_features
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            combined = torch.cat([rgb_features, ms_features], dim=1)
            attention_weights = self.attention(combined)
            fused_features = rgb_features * attention_weights[:, 0:1] + \
                           ms_features * attention_weights[:, 1:2]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Final classification
        return self.classifier(fused_features)
    
    def _init_ms_backbone(self):
        """Initialize MS backbone with RGB backbone weights if available."""
        if not hasattr(self, 'rgb_backbone') or not hasattr(self, 'ms_backbone'):
            return
            
        # Copy weights from RGB to MS backbone
        rgb_state_dict = self.rgb_backbone.state_dict()
        ms_state_dict = self.ms_backbone.state_dict()
        
        # Filter out incompatible keys
        filtered_state_dict = {k: v for k, v in rgb_state_dict.items() 
                             if k in ms_state_dict and v.size() == ms_state_dict[k].size()}
        
        # Update MS backbone with compatible weights
        ms_state_dict.update(filtered_state_dict)
        self.ms_backbone.load_state_dict(ms_state_dict)
        
        # Optionally freeze the MS backbone
        if not self.training:
            for param in self.ms_backbone.parameters():
                param.requires_grad = False
    
    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        super().on_train_epoch_start()
        
        # Unfreeze MS backbone after some epochs if needed
        if self.use_ms and hasattr(self, 'ms_backbone') and self.current_epoch >= 5:
            for param in self.ms_backbone.parameters():
                param.requires_grad = True
            # This line is unreachable due to the early return above
            # when MS data is missing but use_ms is True
                                          # The logic above assumes ms_data is present.

        # The final classification step.
        logits = self.classifier(fused_features)
        return logits

    # NO training_step, validation_step, or configure_optimizers needed here!
    # They are all inherited from the powerful BaseModel.