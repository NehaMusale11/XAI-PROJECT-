"""
Partial Dependence Plot (PDP) & Individual Conditional Expectation (ICE) Engine.
Computes marginal response curves showing feature effect on black-box predictions.
"""

import numpy as np
import pandas as pd


class PDPExplainerEngine:
    """Computes Partial Dependence and ICE curves for black-box models."""

    def __init__(self, model, training_data: pd.DataFrame):
        self.model = model
        self.training_data = training_data
        self.feature_names = list(training_data.columns)

    def compute_pdp_ice(self, feature_name: str, num_grid_points: int = 25, sample_size: int = 100):
        """Computes PDP curve and sample ICE curves for a target feature."""
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in training data features.")

        # Downsample dataset for fast computation
        if len(self.training_data) > sample_size:
            sample_df = self.training_data.sample(n=sample_size, random_state=42).copy()
        else:
            sample_df = self.training_data.copy()

        feat_vals = self.training_data[feature_name].values
        min_val, max_val = np.percentile(feat_vals, 2), np.percentile(feat_vals, 98)
        
        if len(np.unique(feat_vals)) <= 10:
            grid_values = np.sort(np.unique(feat_vals))
        else:
            grid_values = np.linspace(min_val, max_val, num_grid_points)

        ice_curves = [] # List of predictions per grid value for each sample instance
        pdp_values = [] # Average prediction per grid value

        grid_values_list = [float(v) for v in grid_values]

        for val in grid_values:
            temp_df = sample_df.copy()
            temp_df[feature_name] = val
            preds = self.model.predict_proba(temp_df)[:, 1]
            pdp_values.append(float(np.mean(preds)))
            ice_curves.append([float(p) for p in preds])

        # Transpose ice_curves to be instance-centric: [instance_idx][grid_idx]
        ice_curves_by_instance = np.array(ice_curves).T.tolist()

        return {
            'feature': feature_name,
            'grid_values': grid_values_list,
            'pdp_values': pdp_values,
            'ice_curves': ice_curves_by_instance[:15] # Return first 15 ICE curves for clean frontend rendering
        }
