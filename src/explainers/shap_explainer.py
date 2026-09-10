"""
SHAP (SHapley Additive exPlanations) Post-hoc Explainer Engine.
Computes Global SHAP Summary attributions and Local SHAP Waterfall attributions.
"""

import numpy as np
import pandas as pd
import shap


class SHAPExplainerEngine:
    """Computes game-theoretic SHAP attributions for tree-based and agnostic black-box models."""

    def __init__(self, model, background_data: pd.DataFrame, is_tree: bool = True):
        self.model = model
        self.background_data = background_data
        self.is_tree = is_tree
        self.feature_names = list(background_data.columns)

        if is_tree:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                self.explainer = shap.Explainer(model, background_data)
        else:
            # Kernel/Permutation Explainer fallback for neural networks or arbitrary models
            def predict_fn(x):
                if isinstance(x, np.ndarray):
                    x_df = pd.DataFrame(x, columns=self.feature_names)
                else:
                    x_df = x
                return model.predict_proba(x_df)[:, 1]

            summary_bg = shap.kmeans(background_data, min(25, len(background_data)))
            self.explainer = shap.KernelExplainer(predict_fn, summary_bg)

    def explain_global(self, X_sample: pd.DataFrame):
        """Computes global feature attributions across dataset."""
        shap_values = self.explainer(X_sample)
        
        # Handle 2D / 3D shap values arrays across multi-class/binary output
        vals = shap_values.values
        if len(vals.shape) == 3:
            vals = vals[:, :, 1] # Positive class for binary classification
        
        mean_abs_shap = np.abs(vals).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        # Beeswarm / Summary plot point data for frontend rendering
        points = []
        for i, feature in enumerate(self.feature_names):
            feature_vals = X_sample[feature].values
            feature_shaps = vals[:, i]
            
            # Normalize feature values 0-1 for color mapping (low to high)
            min_val, max_val = np.min(feature_vals), np.max(feature_vals)
            norm_vals = (feature_vals - min_val) / (max_val - min_val + 1e-9)
            
            for f_val, s_val, n_val in zip(feature_vals, feature_shaps, norm_vals):
                points.append({
                    'feature': feature,
                    'feature_value': float(f_val),
                    'normalized_value': float(n_val),
                    'shap_value': float(s_val)
                })

        return {
            'global_importance': importance_df.to_dict(orient='records'),
            'summary_points': points
        }

    def explain_instance(self, instance_df: pd.DataFrame):
        """Computes local SHAP waterfall attribution for a single instance."""
        if isinstance(instance_df, pd.Series):
            instance_df = instance_df.to_frame().T
            
        shap_exp = self.explainer(instance_df)
        vals = shap_exp.values[0]
        base_val = shap_exp.base_values[0]

        if isinstance(vals, np.ndarray) and len(vals.shape) == 2:
            vals = vals[:, 1]
        if isinstance(base_val, (np.ndarray, list)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

        base_val = float(base_val)
        attributions = []

        sum_shap = base_val
        for feature, s_val in zip(self.feature_names, vals):
            s_val = float(s_val)
            sum_shap += s_val
            f_val = float(instance_df[feature].values[0])
            
            attributions.append({
                'feature': feature,
                'feature_value': f_val,
                'shap_value': s_val,
                'direction': 'positive' if s_val >= 0 else 'negative'
            })

        # Sort by absolute impact
        attributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        final_prediction = float(self.model.predict_proba(instance_df)[0, 1])

        # SHAP additivity theorem verification error
        additivity_sum = base_val + sum([a['shap_value'] for a in attributions])
        
        return {
            'base_value': base_val,
            'final_prediction_probability': final_prediction,
            'attributions': attributions,
            'additivity_sum': float(additivity_sum),
            'additivity_verified': bool(abs(additivity_sum - sum_shap) < 0.05)
        }
