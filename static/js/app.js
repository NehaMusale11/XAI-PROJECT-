/**
 * Post-hoc Explanation of Black-Box Models - Dashboard Logic
 * Handles interactive feature input sliders, API integration, Plotly chart rendering,
 * and explanation comparisons (SHAP, LIME, PDP, Counterfactuals).
 */

document.addEventListener('DOMContentLoaded', async () => {
    // State Management
    let featureNames = [];
    let featureDescriptions = {};
    let presetProfiles = {};
    let currentFeatures = {};
    let selectedModel = 'xgboost';

    // DOM Elements
    const modelSelect = document.getElementById('model-select');
    const slidersContainer = document.getElementById('feature-sliders-container');
    const predictExplainBtn = document.getElementById('predict-explain-btn');
    const runBenchmarkBtn = document.getElementById('run-benchmark-btn');
    const triggerEvalBtn = document.getElementById('trigger-eval-btn');
    const pdpFeatureSelect = document.getElementById('pdp-feature-select');
    const recalcCfBtn = document.getElementById('recalc-cf-btn');

    // 1. Initialize Application Data
    await initAppInfo();

    // 2. Event Listeners setup
    modelSelect.addEventListener('change', (e) => {
        selectedModel = e.target.value;
        runPredictionAndExplanations();
    });

    predictExplainBtn.addEventListener('click', () => {
        runPredictionAndExplanations();
    });

    runBenchmarkBtn.addEventListener('click', () => {
        switchTab('tab-benchmark');
        fetchBenchmarkData();
    });

    triggerEvalBtn.addEventListener('click', () => {
        fetchBenchmarkData();
    });

    pdpFeatureSelect.addEventListener('change', (e) => {
        renderPDPChart(e.target.value);
    });

    recalcCfBtn.addEventListener('click', () => {
        fetchCounterfactuals();
    });

    setupTabNavigation();
    setupPresetButtons();

    // Initial Execution
    await runPredictionAndExplanations();
    await renderGlobalImportance();
    await renderPDPChart(pdpFeatureSelect.value || 'oldpeak');

    /**
     * Fetches metadata from Flask backend and initializes UI sliders
     */
    async function initAppInfo() {
        try {
            const resp = await fetch('/api/info');
            const data = await resp.json();

            featureNames = data.feature_names;
            featureDescriptions = data.feature_descriptions;
            presetProfiles = data.profiles;

            // Load high_risk profile as default initial state
            currentFeatures = { ...presetProfiles.high_risk.features };

            buildFeatureSliders();
            populatePDPDropdown();
            renderModelsEvalTable(data.metrics);
        } catch (err) {
            console.error('Error initializing info:', err);
        }
    }

    /**
     * Builds interactive HTML sliders for all features
     */
    function buildFeatureSliders() {
        slidersContainer.innerHTML = '';

        const featureRanges = {
            'age': { min: 29, max: 77, step: 1 },
            'sex': { min: 0, max: 1, step: 1 },
            'cp': { min: 0, max: 3, step: 1 },
            'trestbps': { min: 94, max: 200, step: 2 },
            'chol': { min: 126, max: 564, step: 5 },
            'fbs': { min: 0, max: 1, step: 1 },
            'restecg': { min: 0, max: 2, step: 1 },
            'thalach': { min: 71, max: 202, step: 2 },
            'exang': { min: 0, max: 1, step: 1 },
            'oldpeak': { min: 0.0, max: 6.2, step: 0.1 },
            'slope': { min: 0, max: 2, step: 1 },
            'ca': { min: 0, max: 4, step: 1 },
            'thal': { min: 0, max: 2, step: 1 }
        };

        featureNames.forEach(feat => {
            const val = currentFeatures[feat] !== undefined ? currentFeatures[feat] : 0;
            const range = featureRanges[feat] || { min: 0, max: 100, step: 1 };
            const desc = featureDescriptions[feat] || feat;

            const group = document.createElement('div');
            group.className = 'slider-group';
            group.title = desc;

            group.innerHTML = `
                <div class="slider-header">
                    <span class="slider-title">${feat.toUpperCase()}</span>
                    <span id="slider-val-${feat}" class="slider-val-badge">${val}</span>
                </div>
                <input type="range" 
                       id="slider-input-${feat}" 
                       class="custom-range" 
                       min="${range.min}" 
                       max="${range.max}" 
                       step="${range.step}" 
                       value="${val}">
            `;

            slidersContainer.appendChild(group);

            const inputElem = group.querySelector(`#slider-input-${feat}`);
            inputElem.addEventListener('input', (e) => {
                const parsedVal = parseFloat(e.target.value);
                currentFeatures[feat] = parsedVal;
                document.getElementById(`slider-val-${feat}`).textContent = parsedVal;
            });
        });
    }

    /**
     * Binds click events to patient preset profile buttons
     */
    function setupPresetButtons() {
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.getAttribute('data-preset');
                if (presetProfiles[key]) {
                    currentFeatures = { ...presetProfiles[key].features };
                    featureNames.forEach(feat => {
                        const inputElem = document.getElementById(`slider-input-${feat}`);
                        const valElem = document.getElementById(`slider-val-${feat}`);
                        if (inputElem && valElem) {
                            inputElem.value = currentFeatures[feat];
                            valElem.textContent = currentFeatures[feat];
                        }
                    });
                    runPredictionAndExplanations();
                }
            });
        });
    }

    /**
     * Tab Navigation setup
     */
    function setupTabNavigation() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                switchTab(targetTab);
            });
        });
    }

    function switchTab(targetTabId) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

        const activeBtn = document.querySelector(`.tab-btn[data-tab="${targetTabId}"]`);
        const activePane = document.getElementById(targetTabId);

        if (activeBtn) activeBtn.classList.add('active');
        if (activePane) activePane.classList.add('active');

        // Trigger chart resize if needed
        window.dispatchEvent(new Event('resize'));
    }

    function populatePDPDropdown() {
        pdpFeatureSelect.innerHTML = '';
        featureNames.forEach(feat => {
            const opt = document.createElement('option');
            opt.value = feat;
            opt.textContent = feat.toUpperCase() + ` (${featureDescriptions[feat] || ''})`;
            if (feat === 'oldpeak') opt.selected = true;
            pdpFeatureSelect.appendChild(opt);
        });
    }

    /**
     * Main Pipeline Execution: Predict & Generate Explanations
     */
    async function runPredictionAndExplanations() {
        const payload = {
            model: selectedModel,
            features: currentFeatures
        };

        // 1. Fetch Model Prediction
        try {
            const predResp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const predData = await predResp.json();

            updatePredictionBanner(predData);

            // 2. Fetch SHAP & LIME local explanations concurrently
            const [shapResp, limeResp] = await Promise.all([
                fetch('/api/explain/shap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }),
                fetch('/api/explain/lime', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                })
            ]);

            const shapData = await shapResp.json();
            const limeData = await limeResp.json();

            renderSHAPWaterfallChart(shapData);
            renderLIMEBarChart(limeData);

            // 3. Fetch Counterfactuals
            await fetchCounterfactuals();

        } catch (err) {
            console.error('Error running predictions/explanations:', err);
        }
    }

    /**
     * Updates top prediction banner UI elements
     */
    function updatePredictionBanner(data) {
        const classBadge = document.getElementById('pred-class-badge');
        const fillElem = document.getElementById('prob-progress-fill');
        const probText = document.getElementById('prob-value-text');
        const wbText = document.getElementById('wb-prob-text');
        const wbDiff = document.getElementById('wb-diff-tag');

        const probPct = (data.prediction_probability * 100).toFixed(1);
        probText.textContent = `${probPct}%`;
        fillElem.style.width = `${probPct}%`;

        if (data.prediction_probability >= 0.5) {
            classBadge.className = 'badge-lg danger';
            classBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> HIGH RISK (${data.prediction_class})`;
        } else {
            classBadge.className = 'badge-lg success';
            classBadge.innerHTML = `<i class="fa-solid fa-shield-heart"></i> LOW RISK (${data.prediction_class})`;
        }

        const wbProbPct = (data.whitebox_comparison.logistic_regression_prob * 100).toFixed(1);
        wbText.textContent = `${wbProbPct}%`;
        wbDiff.textContent = `Diff: ${(data.whitebox_comparison.probability_difference * 100).toFixed(1)}%`;
    }

    /**
     * Renders SHAP Waterfall Chart using Plotly.js
     */
    function renderSHAPWaterfallChart(data) {
        const attributions = data.attributions || [];
        const baseVal = data.base_value;
        const finalVal = data.final_prediction_probability;

        document.getElementById('shap-base-val').textContent = baseVal.toFixed(4);
        document.getElementById('shap-final-val').textContent = finalVal.toFixed(4);

        const addBadge = document.getElementById('shap-additivity-badge');
        if (data.additivity_verified) {
            addBadge.className = 'badge success';
            addBadge.textContent = 'Additivity Verified (Exact Sum)';
        } else {
            addBadge.className = 'badge warning';
            addBadge.textContent = 'Additivity Approximated';
        }

        // Take top 8 features for clean display
        const topAttrs = attributions.slice(0, 8).reverse();

        const yLabels = topAttrs.map(a => `${a.feature.toUpperCase()} = ${a.feature_value}`);
        const xValues = topAttrs.map(a => a.shap_value);
        const colors = topAttrs.map(a => a.shap_value >= 0 ? '#ef4444' : '#10b981');

        const trace = {
            type: 'bar',
            orientation: 'h',
            x: xValues,
            y: yLabels,
            marker: {
                color: colors,
                opacity: 0.85
            },
            text: xValues.map(v => (v >= 0 ? '+' : '') + v.toFixed(4)),
            textposition: 'auto'
        };

        const layout = {
            margin: { l: 120, r: 30, t: 20, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter', color: '#f3f4f6' },
            xaxis: {
                title: 'SHAP Contribution to Prediction Logit / Prob',
                gridcolor: 'rgba(255, 255, 255, 0.08)',
                zerolinecolor: 'rgba(255, 255, 255, 0.3)'
            },
            yaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.08)'
            }
        };

        Plotly.newPlot('shap-waterfall-chart', [trace], layout, { responsive: true, displayModeBar: false });
    }

    /**
     * Renders LIME Sparse Linear Surrogate Bar Chart
     */
    function renderLIMEBarChart(data) {
        const attributions = data.attributions || [];
        const r2Fidelity = data.local_r2_fidelity;

        document.getElementById('lime-r2-badge').textContent = `R² = ${r2Fidelity.toFixed(4)}`;
        document.getElementById('lime-intercept-val').textContent = data.local_intercept.toFixed(4);

        const topAttrs = attributions.slice(0, 8).reverse();

        const yLabels = topAttrs.map(a => `${a.feature.toUpperCase()}`);
        const xValues = topAttrs.map(a => a.weight);
        const colors = topAttrs.map(a => a.weight >= 0 ? '#ef4444' : '#10b981');

        const trace = {
            type: 'bar',
            orientation: 'h',
            x: xValues,
            y: yLabels,
            marker: {
                color: colors,
                opacity: 0.85
            },
            text: xValues.map(v => (v >= 0 ? '+' : '') + v.toFixed(4)),
            textposition: 'auto'
        };

        const layout = {
            margin: { l: 100, r: 30, t: 20, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter', color: '#f3f4f6' },
            xaxis: {
                title: 'LIME Sparse Linear Weight w_i',
                gridcolor: 'rgba(255, 255, 255, 0.08)',
                zerolinecolor: 'rgba(255, 255, 255, 0.3)'
            },
            yaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.08)'
            }
        };

        Plotly.newPlot('lime-bar-chart', [trace], layout, { responsive: true, displayModeBar: false });
    }

    /**
     * Fetches and renders Counterfactual "What-If" Interventions
     */
    async function fetchCounterfactuals() {
        const targetClass = currentFeatures.target_class !== undefined ? currentFeatures.target_class : 0;
        
        try {
            const resp = await fetch('/api/explain/counterfactual', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModel,
                    features: currentFeatures,
                    target_class: targetClass
                })
            });
            const data = await resp.json();

            document.getElementById('cf-orig-prob').textContent = `${(data.original_prediction * 100).toFixed(1)}%`;
            document.getElementById('cf-target-prob').textContent = `${(data.counterfactual_prediction * 100).toFixed(1)}%`;

            const statusBadge = document.getElementById('cf-status-badge');
            if (data.counterfactual_found) {
                statusBadge.className = 'badge success';
                statusBadge.textContent = 'Counterfactual Found (Feasible Solution)';
            } else {
                statusBadge.className = 'badge warning';
                statusBadge.textContent = 'Partial Solution';
            }

            const tbody = document.getElementById('cf-changes-tbody');
            tbody.innerHTML = '';

            if (data.changes && data.changes.length > 0) {
                data.changes.forEach(ch => {
                    const tr = document.createElement('tr');
                    const diffSign = ch.difference >= 0 ? `+${ch.difference}` : `${ch.difference}`;
                    const actionClass = ch.actionable ? 'text-success' : 'text-warning';
                    
                    tr.innerHTML = `
                        <td><strong>${ch.feature.toUpperCase()}</strong></td>
                        <td>${ch.original_value}</td>
                        <td><span class="text-cyan">${ch.counterfactual_value}</span></td>
                        <td><span class="${ch.difference < 0 ? 'text-success' : 'text-danger'}">${diffSign}</span></td>
                        <td><span class="${actionClass}">${ch.actionable ? '<i class="fa-solid fa-check"></i> Actionable' : '<i class="fa-solid fa-lock"></i> Fixed'}</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No feature changes required. Instance already satisfies low-risk threshold.</td></tr>`;
            }

        } catch (err) {
            console.error('Error fetching counterfactuals:', err);
        }
    }

    /**
     * Renders Global Feature Importance Bar Plot
     */
    async function renderGlobalImportance() {
        try {
            const resp = await fetch(`/api/explain/global?model=${selectedModel}`);
            const data = await resp.json();

            const globalImp = data.global_importance || [];
            const sorted = [...globalImp].reverse();

            const yLabels = sorted.map(i => i.feature.toUpperCase());
            const xValues = sorted.map(i => i.mean_abs_shap);

            const trace = {
                type: 'bar',
                orientation: 'h',
                x: xValues,
                y: yLabels,
                marker: {
                    color: '#6366f1',
                    opacity: 0.85
                }
            };

            const layout = {
                margin: { l: 100, r: 30, t: 20, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Inter', color: '#f3f4f6' },
                xaxis: {
                    title: 'Mean |SHAP Value| (Global Importance)',
                    gridcolor: 'rgba(255, 255, 255, 0.08)'
                },
                yaxis: {
                    gridcolor: 'rgba(255, 255, 255, 0.08)'
                }
            };

            Plotly.newPlot('shap-global-bar-chart', [trace], layout, { responsive: true, displayModeBar: false });
        } catch (err) {
            console.error('Error rendering global importance:', err);
        }
    }

    /**
     * Renders Partial Dependence (PDP) & ICE curves
     */
    async function renderPDPChart(featureName) {
        try {
            const resp = await fetch(`/api/explain/pdp?feature=${featureName}&model=${selectedModel}`);
            const data = await resp.json();

            const gridVals = data.grid_values;
            const pdpVals = data.pdp_values;
            const iceCurves = data.ice_curves || [];

            const traces = [];

            // Add sample ICE curves (thin translucent lines)
            iceCurves.forEach((ice, idx) => {
                traces.push({
                    x: gridVals,
                    y: ice,
                    mode: 'lines',
                    line: { color: 'rgba(255, 255, 255, 0.1)', width: 1 },
                    name: idx === 0 ? 'ICE Sample Curves' : '',
                    showlegend: idx === 0
                });
            });

            // Add main PDP curve (bold colored line)
            traces.push({
                x: gridVals,
                y: pdpVals,
                mode: 'lines+markers',
                line: { color: '#06b6d4', width: 3 },
                marker: { size: 6 },
                name: 'Average PDP Curve'
            });

            const layout = {
                margin: { l: 50, r: 30, t: 20, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Inter', color: '#f3f4f6' },
                xaxis: {
                    title: `${featureName.toUpperCase()} Grid Values`,
                    gridcolor: 'rgba(255, 255, 255, 0.08)'
                },
                yaxis: {
                    title: 'Predicted Disease Probability',
                    gridcolor: 'rgba(255, 255, 255, 0.08)',
                    range: [0, 1]
                },
                legend: { orientation: 'h', y: 1.1 }
            };

            Plotly.newPlot('pdp-chart', traces, layout, { responsive: true, displayModeBar: false });
        } catch (err) {
            console.error('Error rendering PDP chart:', err);
        }
    }

    /**
     * Fetches Benchmark fidelity data & updates table
     */
    async function fetchBenchmarkData() {
        try {
            const resp = await fetch('/api/benchmark');
            const data = await resp.json();

            const summary = data.metrics_summary;
            document.getElementById('bm-lime-r2').textContent = summary.avg_lime_r2_fidelity.toFixed(4);
            document.getElementById('bm-shap-error').textContent = summary.avg_shap_additivity_error.toExponential(3);
            document.getElementById('bm-shap-time').textContent = `${summary.latency_ms.shap_avg_ms} ms`;
            document.getElementById('bm-lime-time').textContent = `${summary.latency_ms.lime_avg_ms} ms`;

        } catch (err) {
            console.error('Error fetching benchmark:', err);
        }
    }

    /**
     * Renders performance comparison table for trained models
     */
    function renderModelsEvalTable(metrics) {
        const tbody = document.getElementById('models-eval-tbody');
        tbody.innerHTML = '';

        const typeMap = {
            'xgboost': 'Black-Box (Gradient Boosted Trees)',
            'random_forest': 'Black-Box (Tree Ensemble)',
            'neural_net': 'Black-Box (Multi-Layer Perceptron)',
            'logistic_regression': 'White-Box (Linear Baseline)'
        };

        for (const [name, m] of Object.entries(metrics)) {
            const tr = document.createElement('tr');
            const isWhiteBox = name === 'logistic_regression';
            
            tr.innerHTML = `
                <td><strong>${name.toUpperCase()}</strong></td>
                <td><span class="${isWhiteBox ? 'text-info' : 'text-primary'}">${typeMap[name] || 'Classifier'}</span></td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td>${(m.precision * 100).toFixed(2)}%</td>
                <td>${(m.recall * 100).toFixed(2)}%</td>
                <td><strong>${(m.f1_score * 100).toFixed(2)}%</strong></td>
                <td><span class="text-success">${m.roc_auc.toFixed(4)}</span></td>
            `;
            tbody.appendChild(tr);
        }
    }
});
