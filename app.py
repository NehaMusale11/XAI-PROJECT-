"""
Flask Web Server & REST API for Post-hoc Explanation Dashboard.
Serves prediction endpoints, SHAP, LIME, PDP, Counterfactuals, and benchmark APIs.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from src.data_loader import get_prepared_data, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from src.models import ModelManager
from src.explainers.shap_explainer import SHAPExplainerEngine
from src.explainers.lime_explainer import LIMEExplainerEngine
from src.explainers.pdp_explainer import PDPExplainerEngine
from src.explainers.counterfactual import CounterfactualExplainerEngine
from src.evaluator import PostHocEvaluator

app = Flask(__name__)

# Initialize data and models
print("[App] Loading data and models...")
data_dict = get_prepared_data()
X_train = data_dict['X_train']
X_test = data_dict['X_test']
X_train_scaled = data_dict['X_train_scaled']
X_test_scaled = data_dict['X_test_scaled']
y_train = data_dict['y_train']
y_test = data_dict['y_test']

mm = ModelManager()
models, metrics = mm.load_models()

if not models:
    print("[App] No models found. Training all models...")
    metrics = mm.train_all(
        X_train, y_train, X_test, y_test,
        X_train_scaled, X_test_scaled
    )
    models, _ = mm.load_models()

# Instantiate Explainers for XGBoost as default
shap_engine = SHAPExplainerEngine(models['xgboost'], X_train, is_tree=True)
lime_engine = LIMEExplainerEngine(models['xgboost'], X_train, FEATURE_NAMES)
pdp_engine = PDPExplainerEngine(models['xgboost'], X_train)
cf_engine = CounterfactualExplainerEngine(models['xgboost'], X_train, FEATURE_NAMES)

# Define Sample Patient Profiles
PRESET_PROFILES = {
    'high_risk': {
        'title': 'High Risk Patient (Severe Symptoms)',
        'description': 'Older patient, ST depression 2.8mm, exercise angina, 2 major vessels.',
        'features': {
            'age': 63, 'sex': 1, 'cp': 0, 'trestbps': 145, 'chol': 233, 'fbs': 1,
            'restecg': 0, 'thalach': 120, 'exang': 1, 'oldpeak': 2.8, 'slope': 1, 'ca': 2, 'thal': 2
        }
    },
    'medium_risk': {
        'title': 'Moderate Risk Patient (Borderline)',
        'description': 'Middle-aged patient, mild ST depression, cholesterol 260 mg/dl.',
        'features': {
            'age': 54, 'sex': 1, 'cp': 1, 'trestbps': 132, 'chol': 260, 'fbs': 0,
            'restecg': 1, 'thalach': 148, 'exang': 0, 'oldpeak': 1.2, 'slope': 1, 'ca': 0, 'thal': 1
        }
    },
    'low_risk': {
        'title': 'Low Risk Patient (Healthy Profile)',
        'description': 'Younger patient, high max heart rate 175 bpm, 0 ST depression.',
        'features': {
            'age': 42, 'sex': 0, 'cp': 2, 'trestbps': 118, 'chol': 195, 'fbs': 0,
            'restecg': 0, 'thalach': 175, 'exang': 0, 'oldpeak': 0.0, 'slope': 0, 'ca': 0, 'thal': 0
        }
    }
}


def _get_model(model_name: str):
    return models.get(model_name, models['xgboost'])


def _format_instance(input_json):
    row_dict = {}
    for feat in FEATURE_NAMES:
        val = input_json.get(feat, 0)
        row_dict[feat] = float(val)
    return pd.DataFrame([row_dict], columns=FEATURE_NAMES)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'feature_names': FEATURE_NAMES,
        'feature_descriptions': FEATURE_DESCRIPTIONS,
        'metrics': metrics,
        'profiles': PRESET_PROFILES
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    req = request.get_json()
    model_name = req.get('model', 'xgboost')
    instance_df = _format_instance(req.get('features', {}))

    target_m = _get_model(model_name)
    
    if model_name in ['neural_net', 'logistic_regression']:
        scaled_instance = pd.DataFrame(
            data_dict['scaler'].transform(instance_df),
            columns=FEATURE_NAMES
        )
        prob = float(target_m.predict_proba(scaled_instance)[0, 1])
    else:
        prob = float(target_m.predict_proba(instance_df)[0, 1])

    # White-box baseline prediction comparison
    wb_m = models['logistic_regression']
    scaled_instance = pd.DataFrame(data_dict['scaler'].transform(instance_df), columns=FEATURE_NAMES)
    wb_prob = float(wb_m.predict_proba(scaled_instance)[0, 1])

    return jsonify({
        'model_name': model_name,
        'prediction_probability': prob,
        'prediction_class': int(prob >= 0.5),
        'risk_label': 'High Risk' if prob >= 0.5 else 'Low Risk',
        'whitebox_comparison': {
            'logistic_regression_prob': wb_prob,
            'probability_difference': round(abs(prob - wb_prob), 4)
        }
    })


@app.route('/api/explain/shap', methods=['POST'])
def explain_shap():
    req = request.get_json()
    model_name = req.get('model', 'xgboost')
    instance_df = _format_instance(req.get('features', {}))
    
    target_m = _get_model(model_name)
    engine = SHAPExplainerEngine(target_m, X_train, is_tree=(model_name in ['xgboost', 'random_forest']))
    res = engine.explain_instance(instance_df)
    return jsonify(res)


@app.route('/api/explain/lime', methods=['POST'])
def explain_lime():
    req = request.get_json()
    model_name = req.get('model', 'xgboost')
    instance_df = _format_instance(req.get('features', {}))
    row_series = instance_df.iloc[0]

    target_m = _get_model(model_name)
    engine = LIMEExplainerEngine(target_m, X_train, FEATURE_NAMES)
    res = engine.explain_instance(row_series, num_samples=500)
    return jsonify(res)


@app.route('/api/explain/counterfactual', methods=['POST'])
def explain_counterfactual():
    req = request.get_json()
    model_name = req.get('model', 'xgboost')
    instance_df = _format_instance(req.get('features', {}))
    target_class = int(req.get('target_class', 0))

    target_m = _get_model(model_name)
    engine = CounterfactualExplainerEngine(target_m, X_train, FEATURE_NAMES)
    res = engine.generate_counterfactual(instance_df, target_class=target_class)
    return jsonify(res)


@app.route('/api/explain/pdp', methods=['GET'])
def explain_pdp():
    feature_name = request.args.get('feature', 'oldpeak')
    model_name = request.args.get('model', 'xgboost')
    
    target_m = _get_model(model_name)
    engine = PDPExplainerEngine(target_m, X_train)
    res = engine.compute_pdp_ice(feature_name)
    return jsonify(res)


@app.route('/api/explain/global', methods=['GET'])
def explain_global():
    model_name = request.args.get('model', 'xgboost')
    target_m = _get_model(model_name)
    
    engine = SHAPExplainerEngine(target_m, X_train, is_tree=(model_name in ['xgboost', 'random_forest']))
    res = engine.explain_global(X_test.head(100))
    return jsonify(res)


@app.route('/api/benchmark', methods=['GET'])
def get_benchmark():
    evaluator = PostHocEvaluator(models['xgboost'], X_train, X_test)
    res = evaluator.run_comprehensive_evaluation(n_samples=15)
    return jsonify(res)


if __name__ == '__main__':
    print("[App] Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
