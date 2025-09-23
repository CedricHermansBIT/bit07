---
layout: base.njk
title: "Chapter 2: Linear Regression - Modeling Biological Relationships"
---

<!-- Header -->
<div class="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 2: Linear Regression</h2>
    <p class="text-gray-700 leading-relaxed">Linear regression is a fundamental supervised learning algorithm used for regression tasks. It aims to model the linear relationship between one or more independent variables (features) and a continuous dependent variable (the target). Despite its simplicity, it is a powerful and interpretable tool for understanding relationships in biological data.</p>
</div>

<!-- Core Concepts Section -->
<div class="bg-white rounded-xl shadow-md p-6 mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-6 mt-4 border-b pb-3">The Linear Regression Model</h3>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
            <h4 class="text-xl font-semibold text-gray-700 mb-3">The Hypothesis Function</h4>
            <p class="text-gray-600 mb-4">For a single feature, the model assumes the relationship can be described by a straight line. This is the model's hypothesis:</p>
            <div class="bg-gray-100 rounded-lg p-4 font-mono text-center">
                <code class="text-lg">hθ(x) = θ₀ + θ₁x</code>
            </div>
            <div class="space-y-2 text-sm mt-4">
                <p><code class="font-semibold">hθ(x)</code> is the predicted value.</p>
                <p><code class="font-semibold">x</code> is the input feature (e.g., concentration of a chemical).</p>
                <p><code class="font-semibold">θ₀</code> is the y-intercept (the baseline value when x=0).</p>
                <p><code class="font-semibold">θ₁</code> is the slope, representing the change in the target for a one-unit change in x.</p>
            </div>
            <img src="{{ '/assets/images/linear_regression.png' | url }}" alt="Linear Regression Hypothesis" class="w-half mx-auto">
        </div>
        <div>
            <h4 class="text-xl font-semibold text-gray-700 mb-3">The Cost Function</h4>
            <p class="text-gray-600 mb-4">To find the "best fit" line, we need to quantify how wrong our model is. This is done with a cost function, typically the Mean Squared Error (MSE).</p>
            <div class="bg-gray-100 rounded-lg p-4 font-mono text-center">
                <code class="text-lg">J(θ) = (1/2m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²</code>
            </div>
            <div class="alert-info mt-4">
                <strong>Analogy: A "Disappointment Score"</strong>
                <p>Think of the cost function as a "disappointment score". A high score means the model's predictions are far from the actual data (high disappointment). Our goal is to find the parameters (θ values) that give us the lowest possible disappointment.</p>
            </div>
            <img src="{{ '/assets/images/cost_function.png' | url }}" alt="Cost Function Visualization" class="w-half mx-auto mt-4">
        </div>
    </div>
</div>

<!-- Interactive Demo -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4 mt-4">🔬 Interactive Demo: Adjust the Model Parameters</h3>
    <p class="text-gray-600 mb-4">Adjust the parameters (<code class="text-sm">θ₀</code> and <code class="text-sm">θ₁</code>) to see how the regression line fits the data points.</p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Slope (θ₁)</label>
                    <input type="range" id="slope-slider" min="-2" max="2" step="0.01" value="0.8" class="parameter-slider">
                    <span id="slope-value" class="text-sm text-gray-600">0.8</span>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Intercept (θ₀)</label>
                    <input type="range" id="intercept-slider" min="-5" max="5" step="0.01" value="1.2" class="parameter-slider">
                    <span id="intercept-value" class="text-sm text-gray-600">1.2</span>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Loss</label>
                    <span id="loss-value" class="text-sm text-gray-600">0.0</span>
                </div>
                <button id="regenerate-btn" class="btn-secondary w-full">🔄 Generate New Data</button>
            </div>
        </div>
        <div>
            <div id="regression-plot" style="width:100%;height:250px;"></div>
        </div>
    </div>
</div>


<!-- Gradient Descent Section -->
<div class="bg-white rounded-xl shadow-md p-6 mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4 mt-4">⚙️ Gradient Descent: Finding the Bottom of the Hill</h3>
    <p class="text-gray-700 mb-6">We use a powerful optimization algorithm called Gradient Descent to find the values for θ₀ and θ₁ that minimize the cost function J(θ).</p>
    <div class="highlight">
        <strong>Analogy: The Hiker in the Fog</strong>
        <p>Imagine you are a hiker on a hilly landscape in thick fog. To find the lowest point, you would feel the slope at your feet and take a step in the steepest downhill direction. Repeating this process will eventually lead you to the bottom of the valley. This is exactly how Gradient Descent works.</p>
    </div>
    <div class="mt-4">
        <p class="text-gray-600">It works by repeatedly updating the parameters using the following rule: "The new parameter value is the old value minus a small step (the <code class="text-sm">learning rate α</code>) in the direction of the negative gradient."</p>
    </div>
    <img src="{{ '/assets/images/Gradient_descent.gif' | url }}" alt="Gradient Descent Visualization" class="w-half mx-auto mt-4">
</div>

<!-- Data Pre-processing Section -->
<div class="bg-white rounded-xl shadow-md p-6 mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4 mt-4">🧹 Data Pre-processing: Preparing Data for Modeling</h3>
    <p class="text-gray-700 mb-6">Real-world biological data is rarely "plug-and-play". Proper pre-processing is often more critical for model performance than the choice of algorithm itself.</p>
    <div class="space-y-6">
        <div class="border-l-4 border-blue-400 pl-4">
            <h4 class="font-bold text-gray-800">Feature Scaling</h4>
            <p class="text-gray-600">Transforms features to be on a similar scale, which helps Gradient Descent converge much faster. Common methods include <strong>Standard Scaling</strong> (Z-score), <strong>Min-Max Scaling</strong> and <strong>Robust scaling</strong>.</p>
        </div>
        <div class="border-l-4 border-green-400 pl-4">
            <h4 class="font-bold text-gray-800">Handling Categorical Features</h4>
            <p class="text-gray-600">Models require numerical input. Categorical features (e.g., 'pathway': 'Metabolism', 'Signaling') must be converted. <strong>One-Hot Encoding</strong> is the standard method, creating a new binary column for each category to avoid implying a false ordinal relationship. Another common method is <strong>Label Encoding</strong>, which assigns a unique integer to each category. Note that this will imply an ordinal relationship between the categories.</p>
        </div>
        <div class="border-l-4 border-purple-400 pl-4">
            <h4 class="font-bold text-gray-800">Handling Non-Linearity</h4>
            <p class="text-gray-600">If the relationship isn't a straight line, we can create <strong>polynomial features</strong> (e.g., adding x² as a new feature) to allow the linear model to fit a curve.</p>
        </div>
        <div class="alert-warning mt-6">
            <strong>Critical Concept: Preventing Data Leakage</strong>
            <p>All pre-processing steps (like calculating the mean for scaling) must be learned from the <strong>training set only</strong>. This learned transformation is then applied to both the training and test sets. Fitting on the entire dataset before splitting "leaks" information from the test set into the training process, leading to an overly optimistic performance evaluation.</p>
        </div>
    </div>
</div>

<!-- Bias-Variance and Regularization Section -->
<div class="bg-white rounded-xl shadow-md p-6 mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4 mt-4">🎯 The Bias-Variance Trade-off & Regularization</h3>
    <p class="text-gray-700 mb-4">A central challenge in machine learning is navigating the trade-off between a model that is too simple (<strong>high bias, underfitting</strong>) and one that is too complex and learns the noise in the data (<strong>high variance, overfitting</strong>). This is especially true in bioinformatics, where we often have many features (e.g., genes) and few samples (e.g., patients).</p>
    <p class="text-gray-700 mb-6"><strong>Regularization</strong> is the primary technique to combat overfitting. It works by adding a penalty to the cost function that discourages the model's parameters (θ) from becoming too large.</p>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 class="text-lg font-bold text-blue-800 mb-2">Ridge Regression (L2 Penalty)</h4>
            <p class="text-blue-700">Adds a penalty proportional to the <strong>square</strong> of the coefficients' magnitude. This shrinks large coefficients towards zero but rarely makes them exactly zero. Use this when you believe most features are useful.</p>
        </div>
        <div class="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 class="text-lg font-bold text-green-800 mb-2">Lasso Regression (L1 Penalty)</h4>
            <p class="text-green-700">Adds a penalty proportional to the <strong>absolute value</strong> of the coefficients' magnitude. This can shrink coefficients to be exactly zero, effectively performing automatic <strong>feature selection</strong>. Use this when you suspect many features are irrelevant.</p>
        </div>
        <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h4 class="text-lg font-bold text-purple-800 mb-2">Elastic Net</h4>
            <p class="text-purple-700">A combination of Ridge and Lasso. The penalty is a weighted sum of the two penalties, with a higher weight given to Ridge.</p>
        </div>
    </div>
</div>

<!-- Improved Ridge vs Lasso Visualization Section -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
  <h3 class="text-xl font-bold text-gray-800 mb-4 mt-4">🔍 Interactive: Ridge vs Lasso Regularization</h3>
  <p class="text-gray-700 mb-4">
    See how the total loss J(θ) = MSE(θ) + α·Penalty(θ) changes with the regularization type and strength.
  </p>
<p class="text-gray-700 mb-4">
  The white cross indicates the most optimal model without regularization, the red dot is the most optimal model with regulariztaion.
</p>

  <div class="grid grid-cols-2 md:grid-cols-2 gap-4 mb-4">
    <div class="">
      <div class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Regularization Type</label>
          <div class="flex space-x-4 mb-4">
            <button id="ridge-btn" class="px-4 py-2 bg-blue-500 text-white rounded-md">Ridge (L2)</button>
            <button id="lasso-btn" class="px-4 py-2 bg-gray-200 text-gray-700 rounded-md">Lasso (L1)</button>
          </div>
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Regularization Strength (α)</label>
          <input type="range" id="alpha-slider" min="0" max="5" step="0.1" value="1" class="parameter-slider w-full">
          <span id="alpha-value" class="text-sm text-gray-600">1.0</span>
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">3D Surface</label>
          <div class="flex items-center gap-3 text-sm">
            <label class="inline-flex items-center gap-1">
              <input type="radio" name="surfaceMode" value="total" checked>
              <span>Total</span>
            </label>
            <label class="inline-flex items-center gap-1">
              <input type="radio" name="surfaceMode" value="mse">
              <span>MSE</span>
            </label>
            <label class="inline-flex items-center gap-1">
              <input type="radio" name="surfaceMode" value="reg">
              <span>Regularizer</span>
            </label>
          </div>
        </div>

        <div class="bg-gray-50 p-4 rounded-lg text-sm text-gray-700 space-y-1">
          <div><strong>Unregularized optimum (×):</strong> θ\* = (2, -1)</div>
          <div><strong>Regularized optimum (●):</strong> <span id="opt-coords">θ\_α = (?, ?)</span></div>
          <div id="reg-explanation">
            Ridge: circular constraint regions shrink coefficients proportionally.
          </div>
        </div>
      </div>
    </div>
    <div class="">
      <div id="surface-plot" style="width:100%;height:260px;"></div>
    </div>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <div>
      <div class="text-center text-sm font-semibold mb-1">Mean Squared Error</div>
      <div id="mse-contour" style="width:100%;height:220px;"></div>
    </div>
    <div>
      <div class="text-center text-sm font-semibold mb-1">Regularizer</div>
      <div id="reg-contour" style="width:100%;height:220px;"></div>
    </div>
    <div>
      <div class="text-center text-sm font-semibold mb-1">Total Loss</div>
      <div id="total-contour" style="width:100%;height:220px;"></div>
    </div>
    <div>
      <div class="text-center text-sm font-semibold mb-1">Overlay: MSE + Constraint</div>
      <div id="overlay-contour" style="width:100%;height:220px;"></div>
    </div>
  </div>
</div>


<!-- Multivariate Linear Regression -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4 mt-4">📈 Multivariate Linear Regression</h3>
    <p class="text-gray-700 mb-4">In many biological applications, we have multiple features that influence the target variable. Multivariate linear regression extends the simple linear model to accommodate multiple input features.</p>
    <div class="bg-gray-100 rounded-lg p-4 font-mono text-center mb-4">
        <code class="text-lg">hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ</code>
    </div>
    <p class="text-gray-700 mb-4">Here, each <code class="text-sm">xᵢ</code> represents a different feature (e.g., expression levels of different genes), and each <code class="text-sm">θᵢ</code> is the corresponding coefficient that quantifies its contribution to the prediction.</p>
    <p class="text-gray-700">The principles of cost minimization, gradient descent, and regularization apply similarly in the multivariate case, allowing us to build robust models even with many features.</p>
</div>

<div id="next-steps" class="bg-gradient-to-r from-primary-100 to-secondary      -100 rounded-2xl p-8 text-center">
    <h3 class="text-2xl font-bold gradient-text mb-4">Next Steps</h3>
    <p class="text-gray-700 max-w-3xl mx-auto">You now understand how to build, optimize, and regularize a linear regression model. The next step is to adapt these ideas for classification problems.</p>
    <div class="flex flex-wrap justify-center gap-4 mt-6">
        <a href="{{ '/logistic-regression/' | url }}" class="btn-primary">Next: Logistic Regression →</a>
        <a href="{{ '/introduction/' | url }}" class="btn-secondary">← Previous: Introduction</a>
    </div>
</div>

<script>
    let data = generateData();
    function generateData() {
        const n = 50;
        const x_vals = Array.from({length: n}, () => Math.random() * 10);
        const noise_level = 3;
        const y_vals = x_vals.map(x => 1.5 + 0.5 * x + (Math.random() - 0.5) * noise_level);
        return {x: x_vals, y: y_vals};
    }
    // Interactive Linear Regression Demo
    function updatePlot() {
        const slope = parseFloat(document.getElementById('slope-slider').value);
        const intercept = parseFloat(document.getElementById('intercept-slider').value);
        
        document.getElementById('slope-value').textContent = slope.toFixed(2);
        document.getElementById('intercept-value').textContent = intercept.toFixed(2);

        const x_vals = data.x;
        const y_vals = data.y;
        const n = x_vals.length;
        
        const user_line_x = [0, 10];
        const user_line_y = [intercept, intercept + slope * 10];

        let loss = 0;
        for (let i = 0; i < n; i++) {
            const y_hat = slope * x_vals[i] + intercept;
            const error = y_vals[i] - y_hat;
            loss += error * error;
        }

        // Update loss
        document.getElementById('loss-value').textContent = loss.toFixed(2);
        
        const data_points = {
            x: x_vals,
            y: y_vals,
            mode: 'markers',
            type: 'scatter',
            name: 'Biological Data',
            marker: { color: '#3B82F6', size: 8 }
        };
        
        const user_line = {
            x: user_line_x,
            y: user_line_y,
            mode: 'lines',
            type: 'scatter',
            name: 'Your Model',
            line: { color: '#EF4444', width: 3 }
        };
        
        const layout = {
            title: 'Gene Expression vs. Drug Dose',
            xaxis: { title: 'Drug Dose (μM)' },
            yaxis: { title: 'Gene Expression' },
            margin: { t: 40, r: 20, b: 40, l: 50 },
            showlegend: false,
            font: {size: 10}
        };
        
        Plotly.newPlot('regression-plot', [data_points, user_line], layout);
    }
    
    document.getElementById('slope-slider').addEventListener('input', updatePlot);
    document.getElementById('intercept-slider').addEventListener('input', updatePlot);
    document.getElementById('regenerate-btn').addEventListener('click', () => {
        data = generateData();
        updatePlot();
    });
    
    document.addEventListener('DOMContentLoaded', updatePlot);
</script>


<script>
  // Self-contained, fixes any missing brace issues
  document.addEventListener('DOMContentLoaded', function () {
    // Ground-truth minimum of the simplified MSE: (2, -1)
    const t1 = 2, t2 = -1;

    // State
    let regType = 'ridge';
    let alpha = 1.0;
    let surfaceMode = 'total';

    // Grid
    const n = 60;
    const range = 4;
    const theta1 = Array.from({ length: n }, (_, i) => -range + i * (2 * range) / (n - 1));
    const theta2 = Array.from({ length: n }, (_, i) => -range + i * (2 * range) / (n - 1));

    // Helpers
    function softThreshold(v, lam) {
      const s = Math.sign(v) * Math.max(Math.abs(v) - lam, 0);
      return s;
    }

    // Analytic regularized optimum for this toy problem:
    // MSE(θ) = (θ1 - t1)^2 + (θ2 - t2)^2
    // Ridge: θ = t / (1 + α)
    // Lasso: θ = soft_threshold(t, α/2) coordinate-wise
    function analyticOptimum() {
      if (regType === 'ridge') {
        const denom = 1 + alpha;
        return { th1: t1 / denom, th2: t2 / denom };
      } else {
        const lam = alpha / 2;
        return { th1: softThreshold(t1, lam), th2: softThreshold(t2, lam) };
      }
    }

    function buildLosses() {
      const mse = [];
      const reg = [];
      const total = [];
      for (let i = 0; i < n; i++) {
        const rowM = [], rowR = [], rowT = [];
        for (let j = 0; j < n; j++) {
          const x = theta1[i];
          const y = theta2[j];
          const mseVal = (x - t1) * (x - t1) + (y - t2) * (y - t2);
          let regVal = 0;
          if (regType === 'ridge') {
            regVal = alpha * (x * x + y * y);
          } else {
            regVal = alpha * (Math.abs(x) + Math.abs(y));
          }
          rowM.push(mseVal);
          rowR.push(regVal);
          rowT.push(mseVal + regVal);
        }
        mse.push(rowM);
        reg.push(rowR);
        total.push(rowT);
      }
      return { mse, reg, total };
    }

    function diamondPoints(c) {
      // |x| + |y| = c
      return {
        x: [ c,  0, -c,  0,  c],
        y: [ 0,  c,  0, -c,  0]
      };
    }

    function circlePoints(r, k = 120) {
      const x = [], y = [];
      for (let i = 0; i <= k; i++) {
        const a = i * 2 * Math.PI / k;
        x.push(r * Math.cos(a));
        y.push(r * Math.sin(a));
      }
      return { x, y };
    }

    function update() {
      // Compute losses
      const { mse, reg, total } = buildLosses();

      // Optima
      const opt = analyticOptimum();
      const optText = `θ\_α = (${opt.th1.toFixed(2)}, ${opt.th2.toFixed(2)})`;
      document.getElementById('opt-coords').textContent = optText;

      // Colors
      const blue = 'Blues';
      const orange = 'Oranges';
      const viridis = 'Viridis';

      // Layouts
      const base2D = {
        showlegend: false,
        margin: { t: 20, r: 10, b: 30, l: 40 },
        xaxis: { title: 'θ₁', range: [-range, range], zeroline: false },
        yaxis: { title: 'θ₂', range: [-range, range], zeroline: false },
        font: { size: 10 }
      };

      // MSE contour
      Plotly.newPlot('mse-contour', [
        {
          x: theta1, y: theta2, z: mse, type: 'contour',
          colorscale: blue, contours: { coloring: 'heatmap' }, showscale: false
        },
        {
          x: [t1], y: [t2], type: 'scatter', mode: 'markers',
          marker: { color: 'white', size: 8, symbol: 'x' }, name: 'Unregularized Optimum'
        }
      ], base2D);

      // Regularizer contour
      Plotly.newPlot('reg-contour', [
        {
          x: theta1, y: theta2, z: reg, type: 'contour',
          colorscale: orange, contours: { coloring: 'heatmap' }, showscale: false
        }
      ], base2D);

      // Total contour
      Plotly.newPlot('total-contour', [
        {
          x: theta1, y: theta2, z: total, type: 'contour',
          colorscale: viridis, contours: { coloring: 'heatmap' }, showscale: false
        },
        {
          x: [opt.th1], y: [opt.th2], type: 'scatter', mode: 'markers',
          marker: { color: 'red', size: 8, symbol: 'circle' }, name: 'Regularized Optimum'
        },
        {
          x: [t1], y: [t2], type: 'scatter', mode: 'markers',
          marker: { color: 'white', size: 8, symbol: 'x' }, name: 'Unregularized Optimum'
        }
      ], base2D);

      // Overlay: MSE line contours + regularizer level set + optima
      const overlayTraces = [
        {
          x: theta1, y: theta2, z: mse, type: 'contour',
          colorscale: blue, contours: { coloring: 'lines', showlines: true }, showscale: false
        },
        {
          x: [opt.th1], y: [opt.th2], type: 'scatter', mode: 'markers',
          marker: { color: 'red', size: 8, symbol: 'circle' }, name: 'Regularized Optimum'
        },
        {
          x: [t1], y: [t2], type: 'scatter', mode: 'markers',
          marker: { color: 'black', size: 8, symbol: 'x' }, name: 'Unregularized Optimum'
        }
      ];

      // Add constraint boundary passing through the regularized optimum
      if (regType === 'ridge') {
        const r = Math.hypot(opt.th1, opt.th2);
        const circ = circlePoints(r);
        overlayTraces.push({
          x: circ.x, y: circ.y, type: 'scatter', mode: 'lines',
          line: { color: '#fb923c', width: 2 }, name: 'L2 level set'
        });
        document.getElementById('reg-explanation').textContent =
          'Ridge: circular constraint regions shrink coefficients proportionally.';
      } else {
        const c = Math.abs(opt.th1) + Math.abs(opt.th2);
        const dia = diamondPoints(c);
        overlayTraces.push({
          x: dia.x, y: dia.y, type: 'scatter', mode: 'lines',
          line: { color: '#fb923c', width: 2 }, name: 'L1 level set'
        });
        document.getElementById('reg-explanation').textContent =
          'Lasso: diamond constraint encourages sparsity (exact zeros).';
      }

      Plotly.newPlot('overlay-contour', overlayTraces, base2D);

      // 3D surface (selectable)
      drawSurface(surfaceMode, { mse, reg, total });
    }

    function drawSurface(mode, losses) {
      let z = losses.total, title = '3D: Total Loss';
      if (mode === 'mse') { z = losses.mse; title = '3D: MSE'; }
      if (mode === 'reg') { z = losses.reg; title = '3D: Regularizer'; }

      const data = [{
        x: theta1, y: theta2, z, type: 'surface', colorscale: 'Viridis', showscale: false,
        contours: { z: { show: true, usecolormap: true, project: { z: true } } }
      }];

      const layout = {
        title,
        scene: {
          xaxis: { title: 'θ₁' },
          yaxis: { title: 'θ₂' },
          zaxis: { title: 'Cost' },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
        },
        margin: { t: 30, r: 0, b: 0, l: 0 },
        font: { size: 10 }
      };

      Plotly.newPlot('surface-plot', data, layout);
    }

    // Events
    document.getElementById('ridge-btn').addEventListener('click', function () {
      regType = 'ridge';
      this.className = 'px-4 py-2 bg-blue-500 text-white rounded-md';
      document.getElementById('lasso-btn').className = 'px-4 py-2 bg-gray-200 text-gray-700 rounded-md';
      update();
    });

    document.getElementById('lasso-btn').addEventListener('click', function () {
      regType = 'lasso';
      this.className = 'px-4 py-2 bg-blue-500 text-white rounded-md';
      document.getElementById('ridge-btn').className = 'px-4 py-2 bg-gray-200 text-gray-700 rounded-md';
      update();
    });

    document.getElementById('alpha-slider').addEventListener('input', function () {
      alpha = parseFloat(this.value);
      document.getElementById('alpha-value').textContent = alpha.toFixed(1);
      update();
    });

    document.querySelectorAll('input[name="surfaceMode"]').forEach(r => {
      r.addEventListener('change', (e) => {
        surfaceMode = e.target.value;
        // Redraw surfaces only; contours update is not needed here
        const losses = buildLosses();
        drawSurface(surfaceMode, losses);
      });
    });

    // Initial render
    document.getElementById('alpha-value').textContent = alpha.toFixed(1);
    update();
  });
</script>