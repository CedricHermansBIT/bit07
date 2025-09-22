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
    <h3 class="text-2xl font-bold text-gray-800 mb-6 border-b pb-3">The Linear Regression Model</h3>

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
        </div>
    </div>
</div>

<!-- Gradient Descent Section -->
<div class="bg-white rounded-xl shadow-md p-6 mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">⚙️ Gradient Descent: Finding the Bottom of the Hill</h3>
    <p class="text-gray-700 mb-6">We use a powerful optimization algorithm called Gradient Descent to find the values for θ₀ and θ₁ that minimize the cost function J(θ).</p>
    <div class="highlight">
        <strong>Analogy: The Hiker in the Fog</strong>
        <p>Imagine you are a hiker on a hilly landscape in thick fog. To find the lowest point, you would feel the slope at your feet and take a step in the steepest downhill direction. Repeating this process will eventually lead you to the bottom of the valley. This is exactly how Gradient Descent works.</p>
    </div>
    <div class="mt-4">
        <p class="text-gray-600">It works by repeatedly updating the parameters using the following rule: "The new parameter value is the old value minus a small step (the <code class="text-sm">learning rate α</code>) in the direction of the negative gradient."</p>
    </div>
</div>

<!-- Data Pre-processing Section -->
<div class="bg-white rounded-xl shadow-md p-6 mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">🧹 Data Pre-processing: Preparing Data for Modeling</h3>
    <p class="text-gray-700 mb-6">Real-world biological data is rarely "plug-and-play". Proper pre-processing is often more critical for model performance than the choice of algorithm itself.</p>
    <div class="space-y-6">
        <div class="border-l-4 border-blue-400 pl-4">
            <h4 class="font-bold text-gray-800">Feature Scaling</h4>
            <p class="text-gray-600">Transforms features to be on a similar scale, which helps Gradient Descent converge much faster. Common methods include <strong>Standard Scaling</strong> (Z-score) and <strong>Min-Max Scaling</strong>.</p>
        </div>
        <div class="border-l-4 border-green-400 pl-4">
            <h4 class="font-bold text-gray-800">Handling Categorical Features</h4>
            <p class="text-gray-600">Models require numerical input. Categorical features (e.g., 'pathway': 'Metabolism', 'Signaling') must be converted. <strong>One-Hot Encoding</strong> is the standard method, creating a new binary column for each category to avoid implying a false ordinal relationship.</p>
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
    <h3 class="text-2xl font-bold text-gray-800 mb-4">🎯 The Bias-Variance Trade-off & Regularization</h3>
    <p class="text-gray-700 mb-4">A central challenge in machine learning is navigating the trade-off between a model that is too simple (<strong>high bias, underfitting</strong>) and one that is too complex and learns the noise in the data (<strong>high variance, overfitting</strong>). This is especially true in bioinformatics, where we often have many features (e.g., genes) and few samples (e.g., patients).</p>
    <p class="text-gray-700 mb-6"><strong>Regularization</strong> is the primary technique to combat overfitting. It works by adding a penalty to the cost function that discourages the model's parameters (θ) from becoming too large.</p>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 class="text-lg font-bold text-blue-800 mb-2">Ridge Regression (L2 Penalty)</h4>
            <p class="text-blue-700">Adds a penalty proportional to the <strong>square</strong> of the coefficients' magnitude. This shrinks large coefficients towards zero but rarely makes them exactly zero. Use this when you believe most features are useful.</p>
        </div>
        <div class="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 class="text-lg font-bold text-green-800 mb-2">Lasso Regression (L1 Penalty)</h4>
            <p class="text-green-700">Adds a penalty proportional to the <strong>absolute value</strong> of the coefficients' magnitude. This can shrink coefficients to be exactly zero, effectively performing automatic <strong>feature selection</strong>. Use this when you suspect many features are irrelevant.</p>
        </div>
    </div>
</div>

<!-- Interactive Demo -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4">🔬 Interactive Demo: Adjust the Model Parameters</h3>
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

<!-- Multivariate Linear Regression -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4">📈 Multivariate Linear Regression</h3>
    <p class="text-gray-700 mb-4">In many biological applications, we have multiple features that influence the target variable. Multivariate linear regression extends the simple linear model to accommodate multiple input features.</p>
    <div class="bg-gray-100 rounded-lg p-4 font-mono text-center mb-4">
        <code class="text-lg">hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ</code>
    </div>
    <p class="text-gray-700 mb-4">Here, each <code class="text-sm">xᵢ</code> represents a different feature (e.g., expression levels of different genes), and each <code class="text-sm">θᵢ</code> is the corresponding coefficient that quantifies its contribution to the prediction.</p>
    <p class="text-gray-700">The principles of cost minimization, gradient descent, and regularization apply similarly in the multivariate case, allowing us to build robust models even with many features.</p>
</div>

<div class="bg-gradient-to-r from-primary-100 to-secondary-100 rounded-2xl p-8 text-center">
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