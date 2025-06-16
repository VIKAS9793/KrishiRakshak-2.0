# KrishiRakshak Architecture

## 1. System Architecture

### 1.1 High-Level Architecture

```
KrishiRakshak/
├── Frontend (Mobile App)
│   ├── Image Capture
│   ├── Model Inference
│   └── Results Display
├── Backend
│   ├── Model Server
│   ├── Data Pipeline
│   └── API Layer
└── ML Pipeline
    ├── Data Processing
    ├── Model Training
    └── Model Export
```

### 1.2 System Architecture Diagrams

#### 1.2.1 Main Deployment
```mermaid
graph TD
    A[Farmer's Device] --> B[Offline App]
    B --> C[Camera]
    B --> D[TFLite Model]
    D --> E[Inference]
    E --> F[Results]
    F --> G[Display]
    F --> H[Recommendations]
    F --> I[Local Storage]
    
    I --> J[SQLite]
    I --> K[Cache]
    B --> L[Mobile App]
    B --> M[Gradio Interface]

    style A fill:#2563eb,stroke:#1e40af,stroke-width:3px,color:#ffffff
    style B fill:#dc2626,stroke:#b91c1c,stroke-width:3px,color:#ffffff
    style C fill:#059669,stroke:#047857,stroke-width:3px,color:#ffffff
    style D fill:#7c3aed,stroke:#6d28d9,stroke-width:3px,color:#ffffff
    style E fill:#ea580c,stroke:#c2410c,stroke-width:3px,color:#ffffff
    style F fill:#0891b2,stroke:#0e7490,stroke-width:3px,color:#ffffff
    style G fill:#65a30d,stroke:#4d7c0f,stroke-width:3px,color:#ffffff
    style H fill:#be185d,stroke:#9d174d,stroke-width:3px,color:#ffffff
    style I fill:#1f2937,stroke:#111827,stroke-width:3px,color:#ffffff
    style J fill:#374151,stroke:#1f2937,stroke-width:3px,color:#ffffff
    style K fill:#4b5563,stroke:#374151,stroke-width:3px,color:#ffffff
    style L fill:#7c2d12,stroke:#6b2914,stroke-width:3px,color:#ffffff
    style M fill:#166534,stroke:#14532d,stroke-width:3px,color:#ffffff
```

#### 1.2.2 Gradio Interface Flow
```mermaid
graph TD
    A[User Uploads Image] --> B[Gradio Interface]
    B --> C[Preprocessing]
    C --> D[Model Inference]
    D --> E[Results Generation]
    E --> F[Display Results]
    F --> G[Save to Local]

    style A fill:#2563eb,stroke:#1e40af,stroke-width:3px,color:#ffffff
    style B fill:#dc2626,stroke:#b91c1c,stroke-width:3px,color:#ffffff
    style C fill:#059669,stroke:#047857,stroke-width:3px,color:#ffffff
    style D fill:#7c3aed,stroke:#6d28d9,stroke-width:3px,color:#ffffff
    style E fill:#ea580c,stroke:#c2410c,stroke-width:3px,color:#ffffff
    style F fill:#0891b2,stroke:#0e7490,stroke-width:3px,color:#ffffff
    style G fill:#65a30d,stroke:#4d7c0f,stroke-width:3px,color:#ffffff
```

### 1.3 Technical Stack (Offline-First)

#### 1.3.1 Frontend Technologies
- **Gradio Interface**
  - **Framework**: Gradio
  - **Features**:
    - Image upload
    - Real-time inference
    - Results visualization
    - Local storage
  - **Benefits**:
    - Easy deployment
    - Cross-platform
    - No internet required
    - Lightweight

- **Mobile**:
  - **Android/iOS**: TFLite
  - **Features**:
    - Camera integration
    - Offline inference
    - Local storage
    - Multi-language
  - **Requirements**:
    - Android 5.0+
    - iOS 13.0+

#### 1.3.2 Backend Components
- **Local Server**
  - **Framework**: Python Flask
  - **Features**:
    - Model serving
    - Image processing
    - Result generation
    - Local database
  - **Performance**:
    - Lightweight
    - Fast response
    - Low memory
    - No internet

- **ML Framework**
  - **Core**: PyTorch 2.0+
  - **Mobile**: TFLite 2.10+
  - **Web**: TensorFlow.js 4.0+
  - **Optimization**: INT8 quantization

### 2. Model Architecture

#### 2.1 Base Model (MobileNetV3-Large)

```mermaid
graph TD
    A[Input 224x224] --> B[Conv2D]
    B --> C[MobileNetV3 Base]
    C --> D[Custom Head]
    D --> E[Output 38 classes]
    
    C --> F[Depthwise Conv]
    F --> G[Squeeze-Excitation]
    G --> H[Hard-Swish]
    
    D --> I[1024 Units]
    I --> J[512 Units]
    J --> K[38 Units]

    style A fill:#2563eb,stroke:#1e40af,stroke-width:3px,color:#ffffff
    style B fill:#dc2626,stroke:#b91c1c,stroke-width:3px,color:#ffffff
    style C fill:#7c3aed,stroke:#6d28d9,stroke-width:3px,color:#ffffff
    style D fill:#ea580c,stroke:#c2410c,stroke-width:3px,color:#ffffff
    style E fill:#0891b2,stroke:#0e7490,stroke-width:3px,color:#ffffff
    style F fill:#8b5cf6,stroke:#7c3aed,stroke-width:3px,color:#ffffff
    style G fill:#a855f7,stroke:#9333ea,stroke-width:3px,color:#ffffff
    style H fill:#c084fc,stroke:#a855f7,stroke-width:3px,color:#ffffff
    style I fill:#f97316,stroke:#ea580c,stroke-width:3px,color:#ffffff
    style J fill:#fb923c,stroke:#f97316,stroke-width:3px,color:#ffffff
    style K fill:#fdba74,stroke:#fb923c,stroke-width:3px,color:#000000
```

#### 2.2 Technical Specifications

1. **Base Architecture**
   - **Type**: MobileNetV3 Large
   - **Input Size**: 224x224 RGB images
   - **Output Classes**: 38 plant disease classes
   - **Model Size**: ~5.4MB (FP32) → ~1.3MB (INT8)
   - **Inference Time**: ~200ms on Snapdragon 6xx
   - **Memory Usage**: ~100MB RAM

2. **Key Features**
   - **Efficient Architecture**
     - Depthwise separable convolutions
     - Squeeze-and-excitation blocks
     - Hard-swish activation
     - EfficientNet scaling rules

   - **Transfer Learning**
     - Pre-trained on ImageNet
     - Fine-tuned on PlantVillage
     - Custom head for disease classification

3. **Performance Metrics**
   - **Accuracy**: ~95-97% on PlantVillage
   - **F1 Score**: ~0.94 (weighted)
   - **Precision**: ~0.95
   - **Recall**: ~0.94

4. **Resource Usage**
   - **CPU**: Optimized for mobile devices
   - **Memory**: ~100MB RAM during inference
   - **Storage**: ~1.3MB (quantized)
   - **Battery**: Low power consumption

#### 2.3 Custom Head Architecture

```mermaid
graph TD
    A[Input Features] --> B[1024 Units]
    B --> C[Hard-Swish]
    C --> D[Dropout 0.2]
    D --> E[512 Units]
    E --> F[Hard-Swish]
    F --> G[Dropout 0.1]
    G --> H[38 Units]
    H --> I[Softmax]

    style A fill:#2563eb,stroke:#1e40af,stroke-width:3px,color:#ffffff
    style B fill:#dc2626,stroke:#b91c1c,stroke-width:3px,color:#ffffff
    style C fill:#7c3aed,stroke:#6d28d9,stroke-width:3px,color:#ffffff
    style D fill:#ea580c,stroke:#c2410c,stroke-width:3px,color:#ffffff
    style E fill:#0891b2,stroke:#0e7490,stroke-width:3px,color:#ffffff
    style F fill:#059669,stroke:#047857,stroke-width:3px,color:#ffffff
    style G fill:#be185d,stroke:#9d174d,stroke-width:3px,color:#ffffff
    style H fill:#1f2937,stroke:#111827,stroke-width:3px,color:#ffffff
    style I fill:#65a30d,stroke:#4d7c0f,stroke-width:3px,color:#ffffff
```

#### 2.4 Optimization Techniques

1. **Quantization**
   - **Type**: INT8 quantization
   - **Size Reduction**: 4x
   - **Performance Impact**: ~200ms inference
   - **Accuracy Drop**: <1%

2. **Pruning**
   - **Method**: L1 regularization
   - **Reduction**: 30% parameters
   - **Maintained Accuracy**: >95%

3. **Mixed Precision**
   - **Training**: FP16
   - **Inference**: INT8
   - **Memory**: Reduced by 50%
   - **Speed**: Increased by 2x

#### 1.3.4 Storage Solutions
- **Local Storage**
  - **Database**: SQLite
  - **Cache**: IndexedDB
  - **Features**:
    - Offline-first
    - Local persistence
    - Data backup
    - History tracking
  - **Requirements**:
    - Minimal space
    - Fast access
    - Secure storage
    - Backup capability