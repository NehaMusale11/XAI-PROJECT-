"""
Post-hoc Explanation Fidelity & Evaluation Benchmark Module.
Evaluates local surrogate R² fidelity, SHAP additivity error, computation latency,
and feature ranking correlations across post-hoc techniques.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.explainers.shap_explainer import SHAPExplainerEngine
from src.explainers.lime_explainer import LIMEExplainerEngine
from src.explainers.counterfactual import CounterfactualExplainerEngine


class PostHocEvaluator:
    """Benchmark suite for post-hoc interpretability fidelity and efficiency."""

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = list(X_train.columns)

        self.shap_engine = SHAPExplainerEngine(model, X_train, is_tree=True)
        self.lime_engine = LIMEExplainerEngine(model, X_train, self.feature_names)
        self.cf_engine = CounterfactualExplainerEngine(model, X_train, self.feature_names)

    def run_comprehensive_evaluation(self, n_samples: int = 25):
        """Runs full evaluation on a test sample subset."""
        test_subset = self.X_test.head(n_samples)

        shap_times = []
        lime_times = []
        cf_times = []

        lime_fidelities = []
        shap_additivity_errors = []

        print(f"[PostHocEvaluator] Evaluating {n_samples} test instances...")

        for idx, row in test_subset.iterrows():
            instance_df = row.to_frame().T

            # 1. Benchmark SHAP
            t0 = time.time()
            shap_res = self.shap_engine.explain_instance(instance_df)
            shap_times.append((time.time() - t0) * 1000)
            
            # Check additivity error
            pred_prob = shap_res['final_prediction_probability']
            add_sum = shap_res['additivity_sum']
            shap_additivity_errors.append(abs(pred_prob - add_sum))

            # 2. Benchmark LIME
            t0 = time.time()
            lime_res = self.lime_engine.explain_instance(row, num_samples=300)
            lime_times.append((time.time() - t0) * 1000)
            lime_fidelities.append(lime_res['local_r2_fidelity'])

            # 3. Benchmark Counterfactual
            if pred_prob > 0.5:
                t0 = time.time()
                self.cf_engine.generate_counterfactual(instance_df, target_class=0)
                cf_times.append((time.time() - t0) * 1000)

        # Global feature attributions comparison
        shap_global = self.shap_engine.explain_global(test_subset)
        shap_ranking = [item['feature'] for item in shap_global['global_importance']]

        return {
            'metrics_summary': {
                'avg_lime_r2_fidelity': round(float(np.mean(lime_fidelities)), 4),
                'min_lime_r2_fidelity': round(float(np.min(lime_fidelities)), 4),
                'max_lime_r2_fidelity': round(float(np.max(lime_fidelities)), 4),
                'avg_shap_additivity_error': round(float(np.mean(shap_additivity_errors)), 6),
                'latency_ms': {
                    'shap_avg_ms': round(float(np.mean(shap_times)), 2),
                    'lime_avg_ms': round(float(np.mean(lime_times)), 2),
                    'counterfactual_avg_ms': round(float(np.mean(cf_times)), 2) if cf_times else 0.0
                }
            },
            'global_shap_ranking': shap_ranking,
            'sample_eval_count': n_samples
        }


if __name__ == '__main__':
    from src.data_loader import get_prepared_data
    from src.models import ModelManager

    data = get_prepared_data()
    mm = ModelManager()
    models, _ = mm.load_models()

    evaluator = PostHocEvaluator(models['xgboost'], data['X_train'], data['X_test'])
    results = evaluator.run_comprehensive_evaluation(n_samples=10)
    import json
    print(json.dumps(results, indent=2))
