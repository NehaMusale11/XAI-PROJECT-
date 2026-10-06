# Post-hoc Explanation of Black-Box Models 🧠🔍

An end-to-end Explainable Artificial Intelligence (XAI) framework and interactive Web Application designed to **train complex black-box machine learning models** (XGBoost, Random Forest, Deep Neural Networks) and apply **model-agnostic & model-specific post-hoc explanation techniques** to interpret their predictions.

---

## 📌 Features & Explanation Techniques

### 1. Black-Box Models & White-Box Baseline
- **XGBoost Classifier**: Non-linear gradient boosted decision tree ensemble.
- **Random Forest Classifier**: Deep ensemble of randomized decision trees.
- **Multi-Layer Perceptron (MLP)**: Deep Neural Network with hidden layers (64, 32, 16).
- **Logistic Regression (White-Box Baseline)**: Linear model for interpretability comparison.

### 2. Post-hoc Interpretability Engine
- **SHAP (SHapley Additive exPlanations)**:
  - Global Feature Importance: Mean absolute Shapley values ranking.
  - Local Waterfall Attribution: Decomposes instance prediction into additive feature impacts summing to $f(x) = E[f(X)] + \sum \phi_i$.
  - Additivity Verification: Empirical verification of Shapley efficiency property.
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Local Sparse Linear Surrogate: Fits weighted Ridge linear regression on perturbed sample neighborhoods.
  - Neighborhood Fidelity Evaluation: Computes local $R^2$ surrogate score measuring approximation accuracy.
- **Partial Dependence Plots (PDP) & ICE (Individual Conditional Expectation)**:
  - Marginal response curves revealing non-linear feature relationships across value ranges.
- **Counterfactual "What-If" Interventions**:
  - Optimization-based search for minimal actionable feature modifications required to flip high-risk predictions to low-risk (e.g. "If ST depression is reduced from 2.8mm to 0.8mm, disease risk drops from 88% to 22%").

### 3. Interactive Web Dashboard
- Glassmorphism dark mode UI built with Flask, HTML5, CSS3, and Plotly.js.
- Real-time feature sliders and preset patient profiles (High Risk, Moderate Risk, Low Risk).
- Side-by-side SHAP vs LIME comparison workbench.
- Dynamic PDP curve explorer and benchmark metrics viewer.

---

## 📁 Project Structure

```
c:\Users\s2003\OneDrive\Desktop\XAI TAE\
├── data/
│   └── heart_disease.csv            # Synthetic/Real Medical Tabular Dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset generator & preprocessing pipeline
│   ├── models.py                    # Black-box & white-box model training manager
│   ├── explainers/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py        # SHAP waterfall & global importance engine
│   │   ├── lime_explainer.py        # LIME surrogate engine & R² local fidelity calculator
│   │   ├── pdp_explainer.py         # Partial Dependence & ICE curves generator
│   │   └── counterfactual.py        # Minimal counterfactual perturbation engine
│   └── evaluator.py                 # Explanation fidelity & latency benchmark
├── artifacts/                       # Saved trained model binaries (.pkl) & metrics.json
├── static/
│   ├── css/style.css                # Glassmorphic dark-theme styling
│   └── js/app.js                    # Interactive Plotly dashboard JS script
├── templates/
│   └── index.html                   # Web UI HTML5 template
├── app.py                           # Flask REST API backend server
├── main.py                          # CLI runner for training and batch explanation
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## ⚡ Quick Start & Usage

### 1. Requirements & Setup
Ensure Python 3.9+ is installed:
```bash
py -m pip install -r requirements.txt
```

### 2. Train Models & Run CLI Pipeline
To train all black-box models, run local post-hoc explanations, and evaluate fidelity:
```bash
py main.py --train --eval
```

### 3. Launch Interactive Web Dashboard
To start the Flask REST API server and launch the Web Dashboard:
```bash
py app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🚀 Cloud & Production Deployment

The repository is configured for one-click production deployment across cloud providers:

- **Render.com (Free Web Service)**: Connect repository `NehaMusale11/XAI-PROJECT-` to Render. Render automatically builds using `render.yaml` or `Procfile`.
- **Hugging Face Spaces (Free 16GB RAM)**: Create a Docker space and push the repository.
- **Docker / Docker Compose**:
  ```bash
  docker compose up -d
  ```
- **Local WSGI Production Server**:
  ```bash
  # Windows:
  waitress-serve --listen=0.0.0.0:5000 app:app

  # Linux/macOS:
  gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:5000
  ```

See [DEPLOYMENT.md](file:///c:/Users/s2003/OneDrive/Desktop/XAI%20TAE/DEPLOYMENT.md) for step-by-step instructions.

---

## 📊 Empirical Fidelity & Benchmark Results

| Model Architecture | Accuracy | F1-Score | ROC-AUC | Explanation Method | Local Fidelity / Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | **94.17%** | **0.9412** | **0.9852** | SHAP | Additivity Error: $< 10^{-6}$ |
| **Random Forest** | 93.33% | 0.9322 | 0.9780 | LIME | Local $R^2$ Score: **0.9145** |
| **Neural Network (MLP)** | 91.67% | 0.9153 | 0.9650 | Counterfactual | Feasible Solution Search: 100% |
| **Logistic Regression** (White-Box) | 85.83% | 0.8547 | 0.9240 | Linear Coeffs | Global Coefficient Weighting |

---

## 📐 Mathematical Formulation

### SHAP (Shapley Additive exPlanations)
Shapley value $\phi_i$ for feature $i$:
$$\phi_i(x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

### LIME Local Surrogate Optimization
Finding local weight vector $w$:
$$\arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
where $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$ is the exponential distance kernel.
