"""
Main Command Line Runner for Post-hoc Explanation Project.
Executes training pipeline, model evaluation, and sample post-hoc explanations.
"""

import sys
import argparse
import json
import pandas as pd
from src.data_loader import get_prepared_data, FEATURE_NAMES
from src.models import ModelManager
from src.explainers.shap_explainer import SHAPExplainerEngine
from src.explainers.lime_explainer import LIMEExplainerEngine
from src.explainers.pdp_explainer import PDPExplainerEngine
from src.explainers.counterfactual import CounterfactualExplainerEngine
from src.evaluator import PostHocEvaluator


def main():
    parser = argparse.ArgumentParser(description="Post-hoc Explanation of Black-Box Models Pipeline")
    parser.add_argument('--train', action='store_true', help="Force retrain all models")
    parser.add_argument('--eval', action='store_true', help="Run evaluation benchmark on post-hoc fidelity")
    parser.add_argument('--explain-sample', type=int, default=0, help="Index of test sample instance to explain")
    args = parser.parse_args()

    print("=" * 70)
    print("      POST-HOC EXPLANATION OF BLACK-BOX MODELS PIPELINE")
    print("=" * 70)

    # 1. Data Preparation
    data = get_prepared_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    # 2. Model Management
    mm = ModelManager()
    models, metrics = mm.load_models()

    if args.train or not models:
        print("\n--- Step 1: Training Models ---")
        metrics = mm.train_all(
            X_train, y_train, X_test, y_test,
            data['X_train_scaled'], data['X_test_scaled']
        )
        models, _ = mm.load_models()

    print("\n--- Model Performance Summary ---")
    for model_name, m in metrics.items():
        print(f"  [{model_name.upper():<20}] Accuracy: {m['accuracy']:.4f} | F1-Score: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

    target_model = models['xgboost']

    # 3. Post-hoc Explanation Demo
    print(f"\n--- Step 2: Post-hoc Explanation for Test Instance Index {args.explain_sample} ---")
    sample_row = X_test.iloc[args.explain_sample]
    sample_df = sample_row.to_frame().T
    true_label = int(y_test.iloc[args.explain_sample])
    pred_prob = float(target_model.predict_proba(sample_df)[0, 1])

    print(f"Sample Features: {sample_row.to_dict()}")
    print(f"Ground Truth Class: {true_label} | XGBoost Black-Box Predicted Probability: {pred_prob:.4f}")

    # A. SHAP Explanation
    print("\n[A] Computing Local SHAP Waterfall Attribution...")
    shap_engine = SHAPExplainerEngine(target_model, X_train, is_tree=True)
    shap_res = shap_engine.explain_instance(sample_df)
    print(f"    Base Value E[f(X)]: {shap_res['base_value']:.4f}")
    print("    Top 5 SHAP Feature Contributions:")
    for attr in shap_res['attributions'][:5]:
        direction_sign = "+" if attr['direction'] == 'positive' else "-"
        print(f"      - {attr['feature']:<12} (val={attr['feature_value']:>6.1f}): {direction_sign}{abs(attr['shap_value']):.4f}")
    print(f"    SHAP Additivity Verified: {shap_res['additivity_verified']}")

    # B. LIME Explanation
    print("\n[B] Computing Local LIME Sparse Linear Surrogate...")
    lime_engine = LIMEExplainerEngine(target_model, X_train, FEATURE_NAMES)
    lime_res = lime_engine.explain_instance(sample_row, num_samples=500)
    print(f"    LIME Local Surrogate R² Fidelity Score: {lime_res['local_r2_fidelity']:.4f}")
    print("    Top 5 LIME Feature Weights:")
    for attr in lime_res['attributions'][:5]:
        direction_sign = "+" if attr['direction'] == 'positive' else "-"
        print(f"      - {attr['feature']:<12}: {direction_sign}{abs(attr['weight']):.4f}")

    # C. Counterfactual Explanation
    print("\n[C] Generating Minimal Counterfactual 'What-If' Explanation...")
    cf_engine = CounterfactualExplainerEngine(target_model, X_train, FEATURE_NAMES)
    target_class = 0 if pred_prob > 0.5 else 1
    cf_res = cf_engine.generate_counterfactual(sample_df, target_class=target_class)
    print(f"    Original Prob: {cf_res['original_prediction']:.4f} -> Counterfactual Prob: {cf_res['counterfactual_prediction']:.4f}")
    print("    Recommended Actionable Feature Changes:")
    for ch in cf_res['changes']:
        print(f"      * {ch['feature']}: {ch['original_value']} --> {ch['counterfactual_value']} (Diff: {ch['difference']:+.2f})")

    # 4. Run Evaluation Benchmark if requested
    if args.eval:
        print("\n--- Step 3: Running Fidelity & Evaluation Benchmark ---")
        evaluator = PostHocEvaluator(target_model, X_train, X_test)
        eval_res = evaluator.run_comprehensive_evaluation(n_samples=15)
        print(json.dumps(eval_res['metrics_summary'], indent=4))

    print("\n[Pipeline Complete] To start the Web UI, run: py app.py")


if __name__ == '__main__':
    main()
