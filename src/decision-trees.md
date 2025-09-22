---
layout: base.njk
title: "Chapter 7: Decision Trees"
---

<!-- Header -->
<div class="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 7: Decision Trees</h2>
    <p class="text-gray-700 leading-relaxed">Decision Trees are among the most intuitive and interpretable models in machine learning. They make predictions by learning a hierarchy of simple "if/then/else" questions based on the features in the data, creating a model that is easy to visualize and understand.</p>
</div>

<!-- 1. Tree Structure -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">The Structure of a Decision Tree</h3>
    <p class="text-gray-700 mb-4">A decision tree is structured like an upside-down tree and has several key components:</p>
    <img src="https://techeasyblog.com/wp-content/uploads/2024/06/decision-tree.png" alt="Diagram of a decision tree structure" class="w-full max-w-2xl mx-auto rounded-lg border my-4">
    <ul class="list-disc list-inside space-y-2 text-gray-700">
        <li><strong>Root Node:</strong> The top-most node, which represents the entire dataset before any splits are made.</li>
        <li><strong>Internal Nodes:</strong> Nodes that represent a question or a test on a specific feature (e.g., "Is Gene A Expression > 5.5?").</li>
        <li><strong>Branches:</strong> The links connecting nodes, representing the outcome of a test (e.g., "True" or "False").</li>
        <li><strong>Leaf Nodes (Terminal Nodes):</strong> The final nodes that do not split further. They represent the final class prediction.</li>
    </ul>
</div>

<!-- 2. How a Tree is Built -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">How a Decision Tree is Built: Finding the Best Split</h3>
    <p class="text-gray-700 mb-4">The core of the algorithm is to find the best feature and the best split point to divide the data at each node. The "best" split is the one that makes the resulting child nodes as <strong>"pure"</strong> as possible—meaning they contain samples predominantly from a single class. This purity is measured using criteria like <strong>Gini Impurity</strong> or <strong>Entropy</strong>.</p>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: Find the Best Split</h4>
    <p class="text-gray-600 mb-4">You are the algorithm! Your goal is to find the split that results in the lowest <strong>Weighted Gini Impurity</strong> or the highest <strong>Information Gain</strong>. Try splitting on different features and values.</p>

    <div class="interactive-demo grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Controls -->
        <div class="space-y-4">
            <div>
                <label class="block text-sm font-medium">1. Choose Split Feature:</label>
                <select id="split-feature-select" class="w-full p-2 border rounded mt-1">
                    <option value="gene_a">Gene A Expression (Horizontal Split)</option>
                    <option value="gene_b">Gene B Methylation (Vertical Split)</option>
                </select>
            </div>
            <div>
                <label class="block text-sm font-medium">2. Choose Split Value:</label>
                <input id="split-value-slider" type="range" min="0" max="10" step="0.1" value="5" class="parameter-slider">
                <span id="split-value-display" class="text-sm">Value: 5.0</span>
            </div>
             <div>
                <label class="block text-sm font-medium">3. Choose Criterion:</label>
                <div class="flex gap-4 mt-1">
                    <label class="flex items-center gap-1"><input type="radio" name="criterion" value="gini" checked> Gini</label>
                    <label class="flex items-center gap-1"><input type="radio" name="criterion" value="entropy"> Entropy</label>
                </div>
            </div>
            <div class="text-center">
                <button id="find-best-split-btn" class="btn-secondary w-full">🔍 Show Me the Best Split</button>
            </div>
        </div>
        <!-- Plot -->
        <div>
            <div id="split-plot" style="width:100%; height:300px;"></div>
        </div>
        <!-- Calculation -->
        <div class="lg:col-span-2 bg-gray-50 p-4 rounded-lg font-mono text-xs">
            <h5 class="text-sm font-sans font-bold mb-2 text-center">Split Calculation</h5>
            <div id="calculation-breakdown"></div>
        </div>
    </div>
</div>

<!-- 3. Overfitting and Solutions -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">The Biggest Weakness: Overfitting</h3>
    <p class="text-gray-700 mb-4">If a tree is allowed to grow to its maximum depth, it can create a complex set of rules that perfectly memorizes the training data, including its noise. This model will fail to generalize to new, unseen data. This is <strong>overfitting</strong>.</p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
            <h4 class="text-lg font-semibold text-gray-700">Solution 1: Pruning (Pre-pruning)</h4>
            <p class="text-sm text-gray-600 mb-2">We can stop the tree from growing too deep by setting stopping conditions. This is controlled by key hyperparameters:</p>
            <ul class="list-disc list-inside text-sm space-y-1">
                <li><code>max_depth</code>: Limits the maximum number of levels in the tree.</li>
                <li><code>min_samples_split</code>: The minimum number of samples a node must have to be considered for splitting.</li>
                <li><code>min_samples_leaf</code>: The minimum number of samples that must be in a final leaf node.</li>
            </ul>
        </div>
        <div>
            <h4 class="text-lg font-semibold text-gray-700">Solution 2: Ensembles (Random Forests)</h4>
            <p class="text-sm text-gray-600 mb-2">A single decision tree can be unstable. A more powerful and robust approach is to use a <strong>Random Forest</strong>.</p>
            <div class="alert-info text-sm">A Random Forest is an <strong>ensemble</strong> of many decision trees. It works by training each tree on a different random subset of the data and features. To make a prediction, it aggregates the "votes" from all the individual trees. This process reduces overfitting and leads to much better performance.</div>
        </div>
    </div>
</div>

<!-- Advantages & Disadvantages -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
    <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6">
        <h2 class="text-xl font-bold text-green-800 mb-4">✅ Advantages</h2>
        <ul class="space-y-2 text-green-900">
            <li class="flex items-start gap-2"><strong class="text-green-600">✓</strong> <strong>High Interpretability:</strong> The tree structure is easy to visualize and translate into human-readable rules.</li>
            <li class="flex items-start gap-2"><strong class="text-green-600">✓</strong> <strong>Handles Mixed Data:</strong> Works with both numerical and categorical features with little pre-processing.</li>
            <li class="flex items-start gap-2"><strong class="text-green-600">✓</strong> <strong>Non-linear Relationships:</strong> Can capture complex relationships that linear models miss.</li>
        </ul>
    </div>
    <div class="bg-gradient-to-br from-red-50 to-red-100 rounded-xl p-6">
        <h2 class="text-xl font-bold text-red-800 mb-4">⚠️ Disadvantages</h2>
        <ul class="space-y-2 text-red-900">
            <li class="flex items-start gap-2"><strong class="text-red-600">✗</strong> <strong>Prone to Overfitting:</strong> A single deep tree will likely memorize noise in the training data.</li>
            <li class="flex items-start gap-2"><strong class="text-red-600">✗</strong> <strong>Instability:</strong> Small changes in the data can lead to a completely different tree structure.</li>
        </ul>
    </div>
</div>


<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- INTERACTIVE SPLIT FINDER DEMO ---
    const featureSelect = document.getElementById('split-feature-select');
    const valueSlider = document.getElementById('split-value-slider');
    const valueDisplay = document.getElementById('split-value-display');
    const criterionRadios = document.querySelectorAll('input[name="criterion"]');
    const plotDiv = document.getElementById('split-plot');
    const calcDiv = document.getElementById('calculation-breakdown');
    const findBestBtn = document.getElementById('find-best-split-btn');
    
    let data;

    const generateDemoData = () => {
        data = { gene_a: [], gene_b: [], labels: [] };
        // Class 0
        for (let i = 0; i < 50; i++) {
            data.gene_a.push(Math.random() * 4 + 1); // 1-5
            data.gene_b.push(Math.random() * 5 + 4); // 4-9
            data.labels.push(0);
        }
        // Class 1
        for (let i = 0; i < 50; i++) {
            data.gene_a.push(Math.random() * 5 + 4); // 4-9
            data.gene_b.push(Math.random() * 4 + 1); // 1-5
            data.labels.push(1);
        }
    };

    const calculateImpurity = (labels, criterion) => {
        const n = labels.length;
        if (n === 0) return 0;
        const counts = labels.reduce((acc, label) => {
            acc[label] = (acc[label] || 0) + 1;
            return acc;
        }, {});
        
        let impurity = 0;
        if (criterion === 'gini') {
            impurity = 1;
            for (let k in counts) {
                impurity -= (counts[k] / n) ** 2;
            }
        } else { // entropy
            for (let k in counts) {
                const p = counts[k] / n;
                if (p > 0) {
                    impurity -= p * Math.log2(p);
                }
            }
        }
        return impurity;
    };

    const updateDemo = () => {
        const splitFeature = featureSelect.value;
        const splitValue = parseFloat(valueSlider.value);
        const criterion = document.querySelector('input[name="criterion"]:checked').value;
        
        valueDisplay.textContent = `Value: ${splitValue.toFixed(1)}`;
        
        // Split the data
        const leftIndices = [], rightIndices = [];
        data[splitFeature].forEach((val, i) => {
            if (val <= splitValue) {
                leftIndices.push(i);
            } else {
                rightIndices.push(i);
            }
        });
        
        const parentLabels = data.labels;
        const leftLabels = leftIndices.map(i => data.labels[i]);
        const rightLabels = rightIndices.map(i => data.labels[i]);

        // Calculate impurities
        const parentImpurity = calculateImpurity(parentLabels, criterion);
        const leftImpurity = calculateImpurity(leftLabels, criterion);
        const rightImpurity = calculateImpurity(rightLabels, criterion);
        
        // Calculate final score
        let score, scoreName, resultText;
        if (criterion === 'gini') {
            scoreName = "Weighted Gini Impurity";
            const weightedGini = (leftLabels.length / parentLabels.length) * leftImpurity + 
                                 (rightLabels.length / parentLabels.length) * rightImpurity;
            score = weightedGini;
            resultText = `<strong class="text-blue-600">Goal: Minimize this value.</strong> Lower is better.`;
        } else { // entropy
            scoreName = "Information Gain";
            const weightedEntropy = (leftLabels.length / parentLabels.length) * leftImpurity + 
                                    (rightLabels.length / parentLabels.length) * rightImpurity;
            const infoGain = parentImpurity - weightedEntropy;
            score = infoGain;
            resultText = `<strong class="text-blue-600">Goal: Maximize this value.</strong> Higher is better.`;
        }

        // Update calculation breakdown
        calcDiv.innerHTML = `
            <div class="grid grid-cols-3 gap-2 text-center">
                <div><strong>Parent Node</strong></div>
                <div><strong>Left Child (&lt;= ${splitValue.toFixed(1)})</strong></div>
                <div><strong>Right Child (&gt; ${splitValue.toFixed(1)})</strong></div>
                <div>Samples: ${parentLabels.length}</div>
                <div>Samples: ${leftLabels.length}</div>
                <div>Samples: ${rightLabels.length}</div>
                <div>${criterion === 'gini' ? 'Gini' : 'Entropy'}: ${parentImpurity.toFixed(3)}</div>
                <div>${criterion === 'gini' ? 'Gini' : 'Entropy'}: ${leftImpurity.toFixed(3)}</div>
                <div>${criterion === 'gini' ? 'Gini' : 'Entropy'}: ${rightImpurity.toFixed(3)}</div>
            </div>
            <div class="text-center mt-2 pt-2 border-t">
                <strong class="text-base">${scoreName}: <span class="text-lg text-purple-600">${score.toFixed(4)}</span></strong><br>
                <span class="text-xs">${resultText}</span>
            </div>
        `;

        // Update plot
        const traces = [{
            x: data.gene_b, y: data.gene_a, mode: 'markers', type: 'scatter',
            marker: { color: data.labels.map(l => l === 1 ? '#8b5cf6' : '#f59e0b'), size: 8, opacity: 0.8 }
        }];
        
        const splitLine = {
            mode: 'lines', line: { color: '#ef4444', width: 3, dash: 'dash' }
        };
        if (splitFeature === 'gene_a') { // horizontal
            splitLine.x = [0, 10];
            splitLine.y = [splitValue, splitValue];
        } else { // vertical
            splitLine.x = [splitValue, splitValue];
            splitLine.y = [0, 10];
        }
        traces.push(splitLine);
        
        const layout = {
            title: 'Splitting Patient Data',
            xaxis: { title: 'Gene B Methylation', range: [0, 10] },
            yaxis: { title: 'Gene A Expression', range: [0, 10] },
            showlegend: false,
            margin: { t: 30, r: 10, b: 40, l: 50 }
        };
        Plotly.newPlot(plotDiv, traces, layout);
    };

    const findBestSplit = () => {
        let bestScore = (document.querySelector('input[name="criterion"]:checked').value === 'gini') ? Infinity : -Infinity;
        let bestFeature, bestValue;

        ['gene_a', 'gene_b'].forEach(feature => {
            for (let value = 1; value < 9; value += 0.1) {
                const leftLabels = data.labels.filter((_, i) => data[feature][i] <= value);
                const rightLabels = data.labels.filter((_, i) => data[feature][i] > value);
                
                if (leftLabels.length === 0 || rightLabels.length === 0) continue;

                const parentImpurity = calculateImpurity(data.labels, document.querySelector('input[name="criterion"]:checked').value);
                const leftImpurity = calculateImpurity(leftLabels, document.querySelector('input[name="criterion"]:checked').value);
                const rightImpurity = calculateImpurity(rightLabels, document.querySelector('input[name="criterion"]:checked').value);
                
                let currentScore;
                if (document.querySelector('input[name="criterion"]:checked').value === 'gini') {
                    currentScore = (leftLabels.length / data.labels.length) * leftImpurity + (rightLabels.length / data.labels.length) * rightImpurity;
                    if (currentScore < bestScore) {
                        bestScore = currentScore;
                        bestFeature = feature;
                        bestValue = value;
                    }
                } else {
                    const weightedEntropy = (leftLabels.length / data.labels.length) * leftImpurity + (rightLabels.length / data.labels.length) * rightImpurity;
                    currentScore = parentImpurity - weightedEntropy;
                     if (currentScore > bestScore) {
                        bestScore = currentScore;
                        bestFeature = feature;
                        bestValue = value;
                    }
                }
            }
        });
        
        featureSelect.value = bestFeature;
        valueSlider.value = bestValue;
        updateDemo();
    };

    // Event listeners
    [featureSelect, valueSlider, ...criterionRadios].forEach(el => el.addEventListener('input', updateDemo));
    findBestBtn.addEventListener('click', findBestSplit);
    
    // Initial setup
    generateDemoData();
    updateDemo();
});
</script>