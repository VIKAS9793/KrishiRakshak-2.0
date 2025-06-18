#!/usr/bin/env python3
"""
ML/DL Auto Code Review Script
Comprehensive code review tool for Machine Learning and Data Science projects
following best Software Engineering practices.
"""

import ast
import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import subprocess


@dataclass
class CodeIssue:
    """Represents a code issue found during review."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str = ""


class MLCodeReviewer:
    """Main class for ML/DL code review."""
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
        self.ml_imports = {
            'tensorflow', 'tf', 'keras', 'torch', 'pytorch', 'sklearn', 
            'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy', 
            'matplotlib', 'seaborn', 'plotly', 'scipy', 'statsmodels'
        }
        self.security_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'pickle\.loads?\s*\(',
            r'subprocess\.call\s*\(',
            r'os\.system\s*\(',
            r'input\s*\(',
        ]
        
    def review_file(self, file_path: str) -> None:
        """Review a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Syntax error: {e.msg}",
                    suggestion="Fix syntax error before proceeding"
                ))
                return
            
            # Run various checks
            self._check_imports(tree, file_path)
            self._check_ml_best_practices(tree, file_path, content)
            self._check_data_handling(tree, file_path)
            self._check_model_practices(tree, file_path)
            self._check_code_quality(tree, file_path, content)
            self._check_security(content, file_path)
            self._check_documentation(tree, file_path, content)
            self._check_reproducibility(tree, file_path)
            self._check_error_handling(tree, file_path)
            
        except Exception as e:
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=0,
                issue_type="file_error",
                severity="error",
                message=f"Error processing file: {str(e)}"
            ))
    
    def _check_imports(self, tree: ast.AST, file_path: str) -> None:
        """Check import statements for best practices."""
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append((alias.name, alias.asname, node.lineno))
            
            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        imports.append((f"{node.module}.{alias.name}", alias.asname, node.lineno))
        
        ImportVisitor().visit(tree)
        
        # Check for wildcard imports
        for imp_name, alias, line_no in imports:
            if '*' in imp_name:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=line_no,
                    issue_type="import_style",
                    severity="warning",
                    message="Avoid wildcard imports",
                    suggestion="Import specific functions/classes instead of using '*'"
                ))
        
        # Check for common ML import aliases
        ml_alias_suggestions = {
            'numpy': 'np',
            'pandas': 'pd',
            'matplotlib.pyplot': 'plt',
            'seaborn': 'sns',
            'tensorflow': 'tf'
        }
        
        for imp_name, alias, line_no in imports:
            base_name = imp_name.split('.')[0]
            if base_name in ml_alias_suggestions and alias != ml_alias_suggestions.get(imp_name):
                if imp_name in ml_alias_suggestions:
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_no,
                        issue_type="import_convention",
                        severity="info",
                        message=f"Consider using standard alias for {imp_name}",
                        suggestion=f"Use 'import {imp_name} as {ml_alias_suggestions[imp_name]}'"
                    ))
    
    def _check_ml_best_practices(self, tree: ast.AST, file_path: str, content: str) -> None:
        """Check ML-specific best practices."""
        
        class MLVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
            
            def visit_Call(self, node):
                # Check for train_test_split without random_state
                if (hasattr(node.func, 'attr') and node.func.attr == 'train_test_split'):
                    has_random_state = any(
                        kw.arg == 'random_state' for kw in node.keywords
                    )
                    if not has_random_state:
                        self.reviewer.issues.append(CodeIssue(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="reproducibility",
                            severity="warning",
                            message="train_test_split should include random_state for reproducibility",
                            suggestion="Add random_state parameter"
                        ))
                
                # Check for model fitting without validation
                if (hasattr(node.func, 'attr') and node.func.attr == 'fit'):
                    # Look for validation_data or validation_split in keras models
                    if not any(kw.arg in ['validation_data', 'validation_split'] for kw in node.keywords):
                        self.reviewer.issues.append(CodeIssue(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="ml_practice",
                            severity="info",
                            message="Consider adding validation data for model training",
                            suggestion="Add validation_data or validation_split parameter"
                        ))
                
                self.generic_visit(node)
        
        MLVisitor(self).visit(tree)
        
        # Check for missing data preprocessing steps
        if 'fit' in content and 'transform' not in content:
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=0,
                issue_type="data_preprocessing",
                severity="info",
                message="Consider if data preprocessing/transformation is needed",
                suggestion="Ensure proper feature scaling, encoding, etc."
            ))
    
    def _check_data_handling(self, tree: ast.AST, file_path: str) -> None:
        """Check data handling best practices."""
        
        class DataVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
            
            def visit_Call(self, node):
                # Check for pandas read operations without error handling
                if (hasattr(node.func, 'attr') and 
                    node.func.attr in ['read_csv', 'read_excel', 'read_json']):
                    
                    # Check if it's wrapped in try-except
                    parent = getattr(node, 'parent', None)
                    in_try_block = False
                    current = node
                    while hasattr(current, 'parent'):
                        if isinstance(current.parent, ast.Try):
                            in_try_block = True
                            break
                        current = current.parent
                    
                    if not in_try_block:
                        self.reviewer.issues.append(CodeIssue(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="error_handling",
                            severity="warning",
                            message="Data loading should include error handling",
                            suggestion="Wrap in try-except block"
                        ))
                
                # Check for missing data validation
                if (hasattr(node.func, 'attr') and node.func.attr == 'dropna'):
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="data_quality",
                        severity="info",
                        message="Consider logging dropped rows count",
                        suggestion="Log the number of rows dropped for audit trail"
                    ))
                
                self.generic_visit(node)
        
        DataVisitor(self).visit(tree)
    
    def _check_model_practices(self, tree: ast.AST, file_path: str) -> None:
        """Check model-specific best practices."""
        
        class ModelVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
                self.has_model_save = False
                self.has_model_eval = False
            
            def visit_Call(self, node):
                # Check for model saving
                if (hasattr(node.func, 'attr') and 
                    node.func.attr in ['save', 'save_model', 'dump', 'joblib.dump']):
                    self.has_model_save = True
                
                # Check for model evaluation
                if (hasattr(node.func, 'attr') and 
                    node.func.attr in ['evaluate', 'score', 'predict', 'cross_val_score']):
                    self.has_model_eval = True
                
                # Check for hardcoded hyperparameters
                if (hasattr(node.func, 'id') and 
                    any(model in str(node.func.id).lower() for model in 
                        ['randomforest', 'svm', 'xgb', 'lgb', 'neural', 'dense'])):
                    
                    if node.keywords:  # Has hyperparameters
                        self.reviewer.issues.append(CodeIssue(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="hyperparameters",
                            severity="info",
                            message="Consider using configuration files for hyperparameters",
                            suggestion="Move hyperparameters to config file or use hyperparameter tuning"
                        ))
                
                self.generic_visit(node)
            
            def visit_Module(self, node):
                self.generic_visit(node)
                
                # After visiting all nodes, check if model practices are followed
                if not self.has_model_save:
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=0,
                        issue_type="model_persistence",
                        severity="info",
                        message="Consider adding model saving functionality",
                        suggestion="Save trained models for reuse and deployment"
                    ))
                
                if not self.has_model_eval:
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=0,
                        issue_type="model_evaluation",
                        severity="warning",
                        message="No model evaluation found",
                        suggestion="Add model evaluation metrics and validation"
                    ))
        
        ModelVisitor(self).visit(tree)
    
    def _check_code_quality(self, tree: ast.AST, file_path: str, content: str) -> None:
        """Check general code quality issues."""
        lines = content.split('\n')
        
        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="code_style",
                    severity="warning",
                    message="Line too long (>120 characters)",
                    suggestion="Break long lines for better readability"
                ))
        
        # Check for magic numbers
        class MagicNumberVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
            
            def visit_Num(self, node):
                # Ignore common non-magic numbers
                if isinstance(node.n, (int, float)) and node.n not in [0, 1, -1, 2]:
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="magic_number",
                        severity="info",
                        message=f"Consider using named constant instead of magic number {node.n}",
                        suggestion="Define constants at module level"
                    ))
        
        MagicNumberVisitor(self).visit(tree)
        
        # Check function complexity (number of lines)
        class FunctionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type="function_complexity",
                        severity="warning",
                        message=f"Function '{node.name}' is too long ({func_lines} lines)",
                        suggestion="Consider breaking into smaller functions"
                    ))
        
        FunctionVisitor().visit(tree)
    
    def _check_security(self, content: str, file_path: str) -> None:
        """Check for security issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern in self.security_patterns:
                if re.search(pattern, line):
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        issue_type="security",
                        severity="error",
                        message=f"Potential security risk: {pattern.replace('\\s*\\(', '')}",
                        suggestion="Review for security implications"
                    ))
            
            # Check for hardcoded credentials
            if re.search(r'(password|api_key|secret|token)\s*=\s*([\'"]).*?\2', line, re.IGNORECASE):
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="security",
                    severity="error",
                    message="Potential hardcoded credentials",
                    suggestion="Use environment variables or config files"
                ))
    
    def _check_documentation(self, tree: ast.AST, file_path: str, content: str) -> None:
        """Check documentation quality."""
        
        class DocVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
            
            def visit_FunctionDef(self, node):
                # Check for docstring
                if not ast.get_docstring(node):
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="documentation",
                        severity="warning",
                        message=f"Function '{node.name}' missing docstring",
                        suggestion="Add comprehensive docstring with parameters and return value"
                    ))
                
                # Check for type hints
                if not node.returns:
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="type_hints",
                        severity="info",
                        message=f"Function '{node.name}' missing return type hint",
                        suggestion="Add return type annotation"
                    ))
            
            def visit_ClassDef(self, node):
                if not ast.get_docstring(node):
                    self.reviewer.issues.append(CodeIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="documentation",
                        severity="warning",
                        message=f"Class '{node.name}' missing docstring",
                        suggestion="Add class docstring describing purpose and usage"
                    ))
        
        DocVisitor(self).visit(tree)
        
        # Check for module docstring
        try:
            tree_docstring = ast.get_docstring(tree)
            if not tree_docstring:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=1,
                    issue_type="documentation",
                    severity="info",
                    message="Module missing docstring",
                    suggestion="Add module-level docstring describing purpose"
                ))
        except:
            pass
    
    def _check_reproducibility(self, tree: ast.AST, file_path: str) -> None:
        """Check for reproducibility best practices."""
        
        class ReproVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
                self.has_seed = False
            
            def visit_Call(self, node):
                # Check for random seed setting
                if (hasattr(node.func, 'attr') and 
                    node.func.attr in ['seed', 'set_seed', 'random_state']):
                    self.has_seed = True
                
                # Check for numpy/tensorflow/torch random operations without seed
                if (hasattr(node.func, 'attr') and 
                    node.func.attr in ['random', 'randn', 'randint', 'shuffle']):
                    if not self.has_seed:
                        self.reviewer.issues.append(CodeIssue(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="reproducibility",
                            severity="warning",
                            message="Random operations should be seeded for reproducibility",
                            suggestion="Set random seed at beginning of script"
                        ))
                
                self.generic_visit(node)
        
        ReproVisitor(self).visit(tree)
    
    def _check_error_handling(self, tree: ast.AST, file_path: str) -> None:
        """Check error handling patterns."""
        
        class ErrorVisitor(ast.NodeVisitor):
            def __init__(self, reviewer):
                self.reviewer = reviewer
                self.file_path = file_path
            
            def visit_Try(self, node):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        self.reviewer.issues.append(CodeIssue(
                            file_path=self.file_path,
                            line_number=handler.lineno,
                            issue_type="error_handling",
                            severity="warning",
                            message="Bare except clause catches all exceptions",
                            suggestion="Catch specific exception types"
                        ))
                
                self.generic_visit(node)
        
        ErrorVisitor(self).visit(tree)
    
    def generate_report(self, output_format: str = 'text') -> str:
        """Generate review report."""
        if output_format == 'json':
            return json.dumps([
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'message': issue.message,
                    'suggestion': issue.suggestion
                }
                for issue in self.issues
            ], indent=2)
        
        # Text format
        report = []
        report.append("=" * 80)
        report.append("ML/DL CODE REVIEW REPORT")
        report.append("=" * 80)
        
        if not self.issues:
            report.append("✅ No issues found!")
            return '\n'.join(report)
        
        # Group by severity
        by_severity = defaultdict(list)
        for issue in self.issues:
            by_severity[issue.severity].append(issue)
        
        for severity in ['error', 'warning', 'info']:
            if severity in by_severity:
                report.append(f"\n{severity.upper()}S ({len(by_severity[severity])}):")
                report.append("-" * 40)
                
                for issue in by_severity[severity]:
                    report.append(f"\n📁 {issue.file_path}:{issue.line_number}")
                    report.append(f"🔍 [{issue.issue_type}] {issue.message}")
                    if issue.suggestion:
                        report.append(f"💡 Suggestion: {issue.suggestion}")
        
        # Summary
        report.append(f"\n\nSUMMARY:")
        report.append(f"Total issues: {len(self.issues)}")
        for severity in ['error', 'warning', 'info']:
            count = len(by_severity[severity])
            if count > 0:
                report.append(f"{severity.capitalize()}s: {count}")
        
        return '\n'.join(report)
    
    def review_directory(self, directory: str, extensions: List[str] = None) -> None:
        """Review all Python files in a directory."""
        if extensions is None:
            extensions = ['.py']
        
        for root, dirs, files in os.walk(directory):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'venv', 'env']]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    self.review_file(file_path)


def main():
    parser = argparse.ArgumentParser(description='ML/DL Auto Code Review Tool')
    parser.add_argument('path', help='File or directory to review')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--extensions', nargs='+', default=['.py'],
                       help='File extensions to review')
    
    args = parser.parse_args()
    
    reviewer = MLCodeReviewer()
    
    if os.path.isfile(args.path):
        reviewer.review_file(args.path)
    elif os.path.isdir(args.path):
        reviewer.review_directory(args.path, args.extensions)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)
    
    report = reviewer.generate_report(args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)
    
    # Exit with error code if issues found
    error_count = sum(1 for issue in reviewer.issues if issue.severity == 'error')
    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()