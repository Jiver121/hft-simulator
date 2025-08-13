"""
Explainable AI for Financial Models

This module provides model interpretability tools including SHAP and LIME
implementations for understanding model predictions and feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Callable
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class BaseExplainer(ABC):
    """Base class for model explainers."""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize base explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        
    @abstractmethod
    def explain_prediction(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Explain individual prediction."""
        pass
    
    @abstractmethod
    def global_importance(self, X: np.ndarray, **kwargs) -> Dict[str, float]:
        """Calculate global feature importance."""
        pass


class SHAPExplainer(BaseExplainer):
    """SHAP-based model explainer."""
    
    def __init__(self, 
                 model, 
                 feature_names: Optional[List[str]] = None,
                 explainer_type: str = 'auto',
                 background_samples: int = 100):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
            explainer_type: Type of SHAP explainer ('tree', 'kernel', 'deep', 'linear', 'auto')
            background_samples: Number of background samples for kernel explainer
        """
        super().__init__(model, feature_names)
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required but not installed")
        
        self.explainer_type = explainer_type
        self.background_samples = background_samples
        self.explainer = None
        self._background_data = None
        
    def _create_explainer(self, X: np.ndarray):
        """Create appropriate SHAP explainer based on model type."""
        
        if self.explainer is not None:
            return
        
        # Store background data for kernel explainer
        if len(X) > self.background_samples:
            indices = np.random.choice(len(X), self.background_samples, replace=False)
            self._background_data = X[indices]
        else:
            self._background_data = X
        
        # Auto-select explainer type
        if self.explainer_type == 'auto':
            # Try to detect model type
            model_name = self.model.__class__.__name__.lower()
            
            if 'tree' in model_name or 'forest' in model_name or 'xgb' in model_name or 'lgb' in model_name:
                self.explainer_type = 'tree'
            elif 'linear' in model_name or 'ridge' in model_name or 'lasso' in model_name:
                self.explainer_type = 'linear'
            elif hasattr(self.model, 'predict_proba'):
                self.explainer_type = 'kernel'
            else:
                self.explainer_type = 'kernel'
        
        # Create explainer
        try:
            if self.explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self._background_data)
            elif self.explainer_type == 'deep':
                self.explainer = shap.DeepExplainer(self.model, self._background_data)
            elif self.explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(self.model.predict, self._background_data)
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
                
            logger.info(f"Created SHAP {self.explainer_type} explainer")
            
        except Exception as e:
            logger.warning(f"Failed to create {self.explainer_type} explainer: {e}")
            # Fallback to kernel explainer
            self.explainer = shap.KernelExplainer(self.model.predict, self._background_data)
            self.explainer_type = 'kernel'
    
    def explain_prediction(self, X: np.ndarray, max_evals: int = 100) -> Dict[str, Any]:
        """
        Explain individual predictions using SHAP values.
        
        Args:
            X: Input samples to explain
            max_evals: Maximum evaluations for kernel explainer
            
        Returns:
            Dictionary containing SHAP values and explanation
        """
        self._create_explainer(X)
        
        try:
            if self.explainer_type == 'kernel':
                shap_values = self.explainer.shap_values(X, nsamples=max_evals)
            else:
                shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case - take first class
                shap_values = shap_values[0]
            
            # Create feature importance dictionary
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
                
            explanations = []
            for i in range(len(X)):
                sample_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values
                
                feature_importance = {}
                for j, shap_val in enumerate(sample_shap):
                    feature_name = self.feature_names[j] if self.feature_names else f'feature_{j}'
                    feature_importance[feature_name] = float(shap_val)
                
                explanations.append({
                    'shap_values': sample_shap,
                    'feature_importance': feature_importance,
                    'expected_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
                })
            
            return {
                'explanations': explanations,
                'shap_values': shap_values,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
    
    def global_importance(self, X: np.ndarray, max_evals: int = 100) -> Dict[str, float]:
        """
        Calculate global feature importance using mean absolute SHAP values.
        
        Args:
            X: Input samples
            max_evals: Maximum evaluations for kernel explainer
            
        Returns:
            Dictionary of feature importance scores
        """
        explanation = self.explain_prediction(X, max_evals=max_evals)
        
        if 'error' in explanation:
            return {}
        
        shap_values = explanation['shap_values']
        
        # Calculate mean absolute SHAP values
        if len(shap_values.shape) > 1:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_shap = np.abs(shap_values)
        
        importance = {}
        for i, importance_score in enumerate(mean_shap):
            feature_name = self.feature_names[i] if self.feature_names else f'feature_{i}'
            importance[feature_name] = float(importance_score)
        
        return importance
    
    def plot_summary(self, X: np.ndarray, max_display: int = 20, plot_type: str = 'dot'):
        """Create SHAP summary plot."""
        
        explanation = self.explain_prediction(X)
        if 'error' in explanation:
            logger.error("Cannot create plot due to explanation error")
            return
        
        try:
            shap_values = explanation['shap_values']
            feature_names = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            
            if plot_type == 'dot':
                shap.summary_plot(shap_values, X, feature_names=feature_names, 
                                max_display=max_display, show=False)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values, X, feature_names=feature_names,
                                max_display=max_display, plot_type='bar', show=False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create SHAP plot: {e}")


class LIMEExplainer(BaseExplainer):
    """LIME-based model explainer."""
    
    def __init__(self, 
                 model, 
                 feature_names: Optional[List[str]] = None,
                 mode: str = 'regression',
                 discretize_continuous: bool = False):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
            mode: 'regression' or 'classification'
            discretize_continuous: Whether to discretize continuous features
        """
        super().__init__(model, feature_names)
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required but not installed")
        
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        self.explainer = None
        
    def _create_explainer(self, X_train: np.ndarray):
        """Create LIME tabular explainer."""
        
        if self.explainer is not None:
            return
        
        self.explainer = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            mode=self.mode,
            discretize_continuous=self.discretize_continuous
        )
        
        logger.info("Created LIME tabular explainer")
    
    def explain_prediction(self, 
                         X: np.ndarray,
                         X_train: np.ndarray,
                         num_features: int = 10,
                         num_samples: int = 1000) -> Dict[str, Any]:
        """
        Explain individual predictions using LIME.
        
        Args:
            X: Input samples to explain
            X_train: Training data for explainer initialization
            num_features: Number of top features to include in explanation
            num_samples: Number of samples to generate for explanation
            
        Returns:
            Dictionary containing LIME explanations
        """
        self._create_explainer(X_train)
        
        try:
            explanations = []
            
            for i in range(len(X)):
                if len(X.shape) == 1:
                    instance = X
                else:
                    instance = X[i]
                
                # Get explanation
                explanation = self.explainer.explain_instance(
                    instance,
                    self.model.predict,
                    num_features=num_features,
                    num_samples=num_samples
                )
                
                # Extract feature importance
                feature_importance = dict(explanation.as_list())
                
                explanations.append({
                    'feature_importance': feature_importance,
                    'explanation_object': explanation,
                    'intercept': explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0
                })
            
            return {
                'explanations': explanations,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}
    
    def global_importance(self, 
                         X: np.ndarray,
                         X_train: np.ndarray,
                         num_features: int = 10,
                         num_samples: int = 1000) -> Dict[str, float]:
        """
        Calculate global feature importance using LIME.
        
        Args:
            X: Input samples
            X_train: Training data
            num_features: Number of features to explain
            num_samples: Number of samples per explanation
            
        Returns:
            Dictionary of feature importance scores
        """
        explanations = self.explain_prediction(X, X_train, num_features, num_samples)
        
        if 'error' in explanations:
            return {}
        
        # Aggregate feature importance across all explanations
        feature_scores = {}
        
        for explanation in explanations['explanations']:
            for feature, importance in explanation['feature_importance'].items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(abs(importance))
        
        # Calculate mean absolute importance
        global_importance = {}
        for feature, scores in feature_scores.items():
            global_importance[feature] = float(np.mean(scores))
        
        return global_importance


class FinancialExplainer:
    """
    Comprehensive explainer for financial models with domain-specific insights.
    """
    
    def __init__(self, 
                 model,
                 feature_names: Optional[List[str]] = None,
                 feature_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize financial explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
            feature_categories: Dictionary mapping category names to feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.feature_categories = feature_categories or {}
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = SHAPExplainer(model, feature_names)
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
        
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = LIMEExplainer(model, feature_names)
            except Exception as e:
                logger.warning(f"Failed to initialize LIME explainer: {e}")
    
    def explain_prediction(self, 
                         X: np.ndarray,
                         X_train: Optional[np.ndarray] = None,
                         method: str = 'auto') -> Dict[str, Any]:
        """
        Explain predictions using available methods.
        
        Args:
            X: Input samples to explain
            X_train: Training data (required for LIME)
            method: Explanation method ('shap', 'lime', 'auto')
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanations = {}
        
        # SHAP explanation
        if method in ['shap', 'auto'] and self.shap_explainer:
            try:
                shap_explanation = self.shap_explainer.explain_prediction(X)
                explanations['shap'] = shap_explanation
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
        
        # LIME explanation
        if method in ['lime', 'auto'] and self.lime_explainer and X_train is not None:
            try:
                lime_explanation = self.lime_explainer.explain_prediction(X, X_train)
                explanations['lime'] = lime_explanation
            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")
        
        # Add financial insights
        explanations['financial_insights'] = self._generate_financial_insights(explanations)
        
        return explanations
    
    def _generate_financial_insights(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate financial domain-specific insights."""
        
        insights = {
            'risk_factors': [],
            'market_drivers': [],
            'technical_signals': [],
            'category_importance': {}
        }
        
        # Extract feature importance from available explanations
        feature_importance = {}
        
        if 'shap' in explanations and 'explanations' in explanations['shap']:
            for explanation in explanations['shap']['explanations']:
                for feature, importance in explanation['feature_importance'].items():
                    feature_importance[feature] = importance
        
        # Categorize important features
        for category, features in self.feature_categories.items():
            category_score = 0
            category_features = []
            
            for feature in features:
                if feature in feature_importance:
                    importance = abs(feature_importance[feature])
                    category_score += importance
                    category_features.append((feature, importance))
            
            if category_score > 0:
                insights['category_importance'][category] = {
                    'total_importance': category_score,
                    'features': sorted(category_features, key=lambda x: x[1], reverse=True)
                }
        
        # Generate specific insights based on feature categories
        self._add_risk_insights(insights, feature_importance)
        self._add_technical_insights(insights, feature_importance)
        self._add_market_insights(insights, feature_importance)
        
        return insights
    
    def _add_risk_insights(self, insights: Dict[str, Any], feature_importance: Dict[str, float]):
        """Add risk-related insights."""
        
        risk_keywords = ['volatility', 'var', 'risk', 'drawdown', 'beta']
        
        for feature, importance in feature_importance.items():
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in risk_keywords):
                if abs(importance) > 0.1:  # Threshold for significant importance
                    direction = "increasing" if importance > 0 else "decreasing"
                    insights['risk_factors'].append({
                        'feature': feature,
                        'importance': importance,
                        'insight': f"{feature} is {direction} the prediction significantly"
                    })
    
    def _add_technical_insights(self, insights: Dict[str, Any], feature_importance: Dict[str, float]):
        """Add technical analysis insights."""
        
        technical_keywords = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'momentum']
        
        for feature, importance in feature_importance.items():
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in technical_keywords):
                if abs(importance) > 0.1:
                    signal_type = "bullish" if importance > 0 else "bearish"
                    insights['technical_signals'].append({
                        'feature': feature,
                        'importance': importance,
                        'signal': signal_type,
                        'insight': f"{feature} is providing a {signal_type} signal"
                    })
    
    def _add_market_insights(self, insights: Dict[str, Any], feature_importance: Dict[str, float]):
        """Add market structure insights."""
        
        market_keywords = ['volume', 'spread', 'imbalance', 'flow', 'depth']
        
        for feature, importance in feature_importance.items():
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in market_keywords):
                if abs(importance) > 0.1:
                    impact = "positive" if importance > 0 else "negative"
                    insights['market_drivers'].append({
                        'feature': feature,
                        'importance': importance,
                        'impact': impact,
                        'insight': f"Market microstructure feature {feature} has {impact} impact"
                    })
    
    def create_explanation_report(self, 
                                X: np.ndarray,
                                X_train: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> str:
        """
        Create comprehensive explanation report.
        
        Args:
            X: Input samples to explain
            X_train: Training data
            save_path: Path to save report
            
        Returns:
            Explanation report as string
        """
        
        explanations = self.explain_prediction(X, X_train)
        
        report = []
        report.append("# Model Explanation Report")
        report.append("=" * 50)
        report.append("")
        
        # Model predictions
        try:
            predictions = self.model.predict(X)
            report.append(f"## Predictions")
            for i, pred in enumerate(predictions):
                report.append(f"Sample {i+1}: {pred:.4f}")
            report.append("")
        except Exception as e:
            report.append(f"Error getting predictions: {e}")
            report.append("")
        
        # Feature importance
        if 'shap' in explanations and 'explanations' in explanations['shap']:
            report.append("## Top Feature Importances (SHAP)")
            
            # Get first explanation for feature importance
            first_explanation = explanations['shap']['explanations'][0]
            feature_importance = first_explanation['feature_importance']
            
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for feature, importance in sorted_features[:10]:
                report.append(f"- {feature}: {importance:.4f}")
            report.append("")
        
        # Financial insights
        if 'financial_insights' in explanations:
            insights = explanations['financial_insights']
            
            if insights['risk_factors']:
                report.append("## Risk Factors")
                for risk in insights['risk_factors']:
                    report.append(f"- {risk['insight']}")
                report.append("")
            
            if insights['technical_signals']:
                report.append("## Technical Signals")
                for signal in insights['technical_signals']:
                    report.append(f"- {signal['insight']}")
                report.append("")
            
            if insights['market_drivers']:
                report.append("## Market Drivers")
                for driver in insights['market_drivers']:
                    report.append(f"- {driver['insight']}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report_text
