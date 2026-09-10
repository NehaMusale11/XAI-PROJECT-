"""
Counterfactual Explanation Engine ("What-If" Minimal Perturbation Search).
Finds minimal actionable feature modifications to flip black-box predictions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class CounterfactualExplainerEngine:
    """Finds sparse, actionable counterfactual instances for black-box models."""

    def __init__(self, model, training_data: pd.DataFrame, feature_names: list):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names

        # Compute feature bounds and standard deviations for distance normalization
        self.feature_mins = np.array(training_data.min().values, copy=True)
        self.feature_maxs = np.array(training_data.max().values, copy=True)
        self.feature_stds = np.array(training_data.std().values, copy=True)
        self.feature_stds[self.feature_stds == 0] = 1.0

        # Define actionable features (e.g. age/sex are fixed, blood pressure/cholesterol/exercise are actionable)
        self.actionable_mask = np.array([
            feat not in ['age', 'sex'] for feat in feature_names
        ])

    def generate_counterfactual(self, instance_df: pd.DataFrame, target_class: int = 0, desired_prob_threshold: float = 0.35):
        """Generates a minimal actionable counterfactual instance to achieve target_class."""
        if isinstance(instance_df, pd.Series):
            instance_df = instance_df.to_frame().T

        x_orig = instance_df.iloc[0].values.astype(float)
        orig_prob = float(self.model.predict_proba(instance_df)[0, 1])

        # If already meeting threshold
        if (target_class == 0 and orig_prob <= desired_prob_threshold) or (target_class == 1 and orig_prob >= desired_prob_threshold):
            return {
                'original_prediction': orig_prob,
                'counterfactual_prediction': orig_prob,
                'counterfactual_found': True,
                'changes': [],
                'message': 'Instance already satisfies target class condition.'
            }

        # Optimization objective: minimize (feature distance + penalty * probability_violation)
        def objective(x):
            # Normalize distance by std dev
            dist = np.sum(np.abs(x - x_orig) / self.feature_stds)
            
            # Non-actionable feature penalty
            non_actionable_penalty = np.sum(np.abs(x[~self.actionable_mask] - x_orig[~self.actionable_mask])) * 1000.0
            
            # Predict probability
            x_curr_df = pd.DataFrame([x], columns=self.feature_names)
            prob = self.model.predict_proba(x_curr_df)[0, 1]

            if target_class == 0:
                prob_penalty = max(0, prob - desired_prob_threshold) * 50.0
            else:
                prob_penalty = max(0, desired_prob_threshold - prob) * 50.0

            return dist + non_actionable_penalty + prob_penalty

        bounds = list(zip(self.feature_mins, self.feature_maxs))

        # Perform L-BFGS-B bounded optimization
        res = minimize(
            objective,
            x0=x_orig.copy(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 250, 'ftol': 1e-4}
        )

        cf_x = res.x
        # Round integer features appropriately
        for i, feat in enumerate(self.feature_names):
            if feat in ['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                cf_x[i] = round(cf_x[i])

        cf_df = pd.DataFrame([cf_x], columns=self.feature_names)
        cf_prob = float(self.model.predict_proba(cf_df)[0, 1])

        # Compute feature differences
        changes = []
        for i, feat in enumerate(self.feature_names):
            orig_val = float(x_orig[i])
            cf_val = float(cf_x[i])
            diff = cf_val - orig_val
            
            if abs(diff) > 1e-3:
                changes.append({
                    'feature': feat,
                    'original_value': round(orig_val, 2),
                    'counterfactual_value': round(cf_val, 2),
                    'difference': round(diff, 2),
                    'actionable': bool(self.actionable_mask[i])
                })

        success = (target_class == 0 and cf_prob <= desired_prob_threshold) or (target_class == 1 and cf_prob >= desired_prob_threshold)

        return {
            'original_prediction': orig_prob,
            'counterfactual_prediction': cf_prob,
            'counterfactual_found': success,
            'changes': changes,
            'counterfactual_features': dict(zip(self.feature_names, [round(v, 2) for v in cf_x]))
        }
