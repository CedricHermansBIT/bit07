---
layout: base.njk
title: "Chapter 8: Ensemble Learning - The Power of Many"
---

<!-- Header -->
<div class="bg-gradient-to-r from-teal-50 to-cyan-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 8: Ensemble Learning - The Power of Many</h2>
    <p class="text-gray-700 leading-relaxed">Instead of relying on a single model, what if we combined the predictions of several? This is the core idea behind **Ensemble Learning**. The central principle is that a diverse group of models, when combined, can achieve better performance and robustness than any single model on its own.</p>
</div>

<!-- 1. The Wisdom of Crowds -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">The Principle: The Wisdom of Crowds</h3>
    <p class="text-gray-600 mb-4">If you have a group of "weak learners" (models that are only slightly better than random chance), you can combine them to create a single "strong learner." This interactive demo shows the probability that a majority vote of multiple models will be correct.</p>
    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <label class="block text-sm font-medium">Accuracy of Each Base Model</label>
            <input id="base-accuracy" type="range" min="0.51" max="0.99" step="0.01" value="0.7" class="parameter-slider">
            <span id="base-accuracy-value" class="text-sm">70%</span>
            <div class="mt-4">
                <p class="text-sm text-gray-700">With <strong class="text-primary-600">25</strong> base models, each with <strong class="text-primary-600" id="base-accuracy-text">70%</strong> accuracy, the ensemble's accuracy through majority vote is:</p>
                <div id="ensemble-accuracy" class="text-4xl font-bold font-mono text-green-600 my-2">97.6%</div>
            </div>
        </div>
        <div>
            <div id="ensemble-plot" style="width:100%;height:250px;"></div>
        </div>
    </div>
</div>

<!-- 2. Bagging -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">1. Bagging: Bootstrap Aggregating</h3>
    <p class="text-gray-700 mb-4">Bagging focuses on **reducing variance** by training the same algorithm on different random subsets of the data and then averaging their predictions. It's a parallel process where models are trained independently.</p>
    <div class="highlight"><strong>Key Algorithm: Random Forest.</strong> This is a specific implementation of bagging where the base model is a Decision Tree. It adds an extra layer of randomness by also selecting a random subset of features at each split.</div>

    <h4 class="text-lg font-semibold mt-6 mb-4">Interactive Demo: Bootstrap Sampling</h4>
    <p class="text-gray-600 mb-4">The core of bagging is **bootstrap sampling**—sampling with replacement. Click the button to create a new "bag" from the original dataset. Notice how some data points are selected multiple times (larger dots), while others are left out (grayed out).</p>
    <div class="interactive-demo">
        <div class="text-center mb-4">
            <button id="generate-bag-btn" class="btn-primary">Generate New Bootstrap Sample ("Bag")</button>
        </div>
        <div id="bagging-plot" class="w-full h-48 border rounded-lg p-2 flex flex-wrap gap-2 items-center justify-center">
            <!-- JS will populate this -->
        </div>
    </div>
</div>

<!-- 3. Boosting -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">2. Boosting: Learning from Mistakes</h3>
    <p class="text-gray-700 mb-4">Boosting trains models **sequentially**, where each new model is trained to correct the errors made by the previous ones. It focuses on turning a collection of weak learners into a single strong learner by focusing on difficult examples.</p>
    <div class="highlight"><strong>Key Algorithms: AdaBoost & Gradient Boosting.</strong> AdaBoost adjusts the weights of misclassified samples. Gradient Boosting trains the next model on the residual errors of the previous one.</div>

    <h4 class="text-lg font-semibold mt-6 mb-4">Interactive Demo: How Boosting Works</h4>
    <p class="text-gray-600 mb-4">Step through the rounds of boosting. In each round, the algorithm increases the weight (size) of the points the previous model misclassified. The new model must then pay more attention to these difficult points.</p>
    <div class="interactive-demo">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div id="boosting-plot" style="width:100%;height:300px;"></div>
            <div class="text-center self-center">
                <h5 id="boosting-round-title" class="text-lg font-bold">Round 1</h5>
                <p id="boosting-description" class="text-sm text-gray-600 mb-4">A simple weak learner (a "stump") splits the data. It misclassifies several points.</p>
                <button id="next-boost-round-btn" class="btn-primary">Next Round →</button>
            </div>
        </div>
    </div>
</div>


<!-- 4. Stacking -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">3. Stacking: The Meta-Learner</h3>
    <p class="text-gray-700 mb-4">Stacking (or Stacked Generalization) takes a different approach. It uses the predictions of several different base models as input features for a final **meta-learner** model. The job of the meta-learner is to learn how to best combine the predictions from the base models.</p>
    <div class="bg-gray-50 p-4 rounded-lg">
        <h4 class="font-semibold text-center mb-2">Stacking Architecture</h4>
        <img src="https://editor.analyticsvidhya.com/uploads/31182An-example-scheme-of-stacking-ensemble-learning%20resized.png" alt="Stacking Architecture Diagram" class="w-full max-w-lg mx-auto">
        <p class="text-xs text-center mt-2"><strong>Level 0:</strong> Diverse base models (e.g., Logistic Regression, Naive Bayes, Decision Tree) make predictions. <br><strong>Level 1:</strong> A meta-learner (often a simple model) takes these predictions as its features to make the final prediction.</p>
    </div>
     <div class="highlight mt-4"><strong>Key Idea:</strong> Stacking works best when the base models are fundamentally different and make different kinds of errors. The meta-learner's job is to learn the strengths and weaknesses of each base model.</div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- 1. WISDOM OF CROWDS DEMO ---
    const baseAccuracySlider = document.getElementById('base-accuracy');
    const baseAccuracyValue = document.getElementById('base-accuracy-value');
    const ensembleAccuracyDisplay = document.getElementById('ensemble-accuracy');
    const ensemblePlotDiv = document.getElementById('ensemble-plot');

    const factorial = n => (n <= 1) ? 1 : n * factorial(n - 1);
    const combinations = (n, k) => factorial(n) / (factorial(k) * factorial(n - k));

    function updateEnsembleAccuracy() {
        const p = parseFloat(baseAccuracySlider.value);
        baseAccuracyValue.textContent = `${(p * 100).toFixed(0)}%`;

        let ensemble_accuracies = [];
        const n_models_range = Array.from({length: 25}, (_, i) => i * 2 + 1);

        n_models_range.forEach(n => {
            let total_prob = 0;
            const k_needed = Math.ceil(n / 2);
            for (let k = k_needed; k <= n; k++) {
                total_prob += combinations(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
            }
            ensemble_accuracies.push(total_prob);
        });
        
        ensembleAccuracyDisplay.textContent = `${(ensemble_accuracies[12] * 100).toFixed(1)}%`; // Display for n=25
        // update base-accuracy-text value
const baseAccuracyText = document.getElementById('base-accuracy-text');
        baseAccuracyText.textContent = `${(p * 100).toFixed(0)}%`;
        
        const trace = {
            x: n_models_range,
            y: ensemble_accuracies,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#10b981' }
        };
        const layout = {
            title: 'Ensemble Accuracy vs. # of Models',
            xaxis: { title: 'Number of Base Models (n)' },
            yaxis: { title: 'Ensemble Accuracy', range: [0.5, 1.0] },
            margin: { t: 30, r: 20, b: 40, l: 50 }
        };
        Plotly.newPlot(ensemblePlotDiv, [trace], layout);
    }
    baseAccuracySlider.addEventListener('input', updateEnsembleAccuracy);
    updateEnsembleAccuracy();

    // --- 2. BAGGING DEMO ---
    const generateBagBtn = document.getElementById('generate-bag-btn');
    const baggingPlotDiv = document.getElementById('bagging-plot');
    const originalDataset = Array.from({length: 20}, (_, i) => i + 1);

    function updateBaggingPlot() {
        const bag = [];
        const counts = {};
        for (let i = 0; i < originalDataset.length; i++) {
            const sample = originalDataset[Math.floor(Math.random() * originalDataset.length)];
            bag.push(sample);
            counts[sample] = (counts[sample] || 0) + 1;
        }

        baggingPlotDiv.innerHTML = '';
        originalDataset.forEach(d => {
            const dot = document.createElement('div');
            dot.textContent = d;
            dot.className = 'w-8 h-8 rounded-full flex items-center justify-center font-bold transition-all';
            if (counts[d]) {
                dot.classList.add('bg-primary-500', 'text-white');
                dot.style.transform = `scale(${1 + counts[d] * 0.2})`;
            } else {
                dot.classList.add('bg-gray-200', 'text-gray-500', 'opacity-50');
            }
            baggingPlotDiv.appendChild(dot);
        });
    }
    generateBagBtn.addEventListener('click', updateBaggingPlot);
    updateBaggingPlot();

    // --- 3. BOOSTING DEMO ---
    const nextBoostBtn = document.getElementById('next-boost-round-btn');
    const boostPlotDiv = document.getElementById('boosting-plot');
    const boostRoundTitle = document.getElementById('boosting-round-title');
    const boostDescription = document.getElementById('boosting-description');
    let boostRound = 0;
    let boostData;

    const generateBoostData = () => ({
        x: Array.from({length: 30}, () => Math.random() * 10),
        y: Array.from({length: 30}, () => Math.random() * 10),
        labels: Array.from({length: 30}, (_,i) => (i < 15 ? 0 : 1)),
        weights: Array(30).fill(1)
    });

    function updateBoostingPlot() {
    const round = boostRound % 3;
    let split_val, split_dim, misclassified;

    if (round === 0) {
        boostRoundTitle.textContent = "Round 1";
        boostDescription.textContent = "A simple weak learner splits the data. Misclassified points are highlighted with a border.";
        split_dim = 'x';
        split_val = 5;
        misclassified = boostData.x.map((x, i) => (x > split_val && boostData.labels[i] === 0) || (x <= split_val && boostData.labels[i] === 1));
    } else if (round === 1) {
        boostRoundTitle.textContent = "Round 2";
        boostDescription.textContent = "The misclassified points from Round 1 are given higher weights (larger size). A new weak learner tries to fix the errors.";
        split_dim = 'y';
        split_val = 6;
        misclassified = boostData.y.map((y, i) => (y > split_val && boostData.labels[i] === 0) || (y <= split_val && boostData.labels[i] === 1));
    } else {
        boostRoundTitle.textContent = "Round 3";
        boostDescription.textContent = "The process repeats. The final model is a weighted combination of all these weak learners.";
        split_dim = 'x';
        split_val = 7;
        misclassified = boostData.x.map((x, i) => (x > split_val && boostData.labels[i] === 0) || (x <= split_val && boostData.labels[i] === 1));
    }
    
    // Create traces for correctly classified and misclassified points separately
    const trace0Correct = {
        x: boostData.x.filter((_, i) => boostData.labels[i] === 0 && !misclassified[i]),
        y: boostData.y.filter((_, i) => boostData.labels[i] === 0 && !misclassified[i]),
        mode: 'markers',
        name: 'Class 0 (Correct)',
        marker: { 
            color: '#60a5fa',
            size: boostData.weights.filter((_, i) => boostData.labels[i] === 0 && !misclassified[i]).map(w => w * 5)
        }
    };
    
    const trace0Wrong = {
        x: boostData.x.filter((_, i) => boostData.labels[i] === 0 && misclassified[i]),
        y: boostData.y.filter((_, i) => boostData.labels[i] === 0 && misclassified[i]),
        mode: 'markers',
        name: 'Class 0 (Misclassified)',
        marker: { 
            color: '#60a5fa',
            size: boostData.weights.filter((_, i) => boostData.labels[i] === 0 && misclassified[i]).map(w => w * 5),
            line: { color: '#ef4444', width: 2 }
        }
    };
    
    const trace1Correct = {
        x: boostData.x.filter((_, i) => boostData.labels[i] === 1 && !misclassified[i]),
        y: boostData.y.filter((_, i) => boostData.labels[i] === 1 && !misclassified[i]),
        mode: 'markers',
        name: 'Class 1 (Correct)',
        marker: { 
            color: '#f97316',
            size: boostData.weights.filter((_, i) => boostData.labels[i] === 1 && !misclassified[i]).map(w => w * 5)
        }
    };
    
    const trace1Wrong = {
        x: boostData.x.filter((_, i) => boostData.labels[i] === 1 && misclassified[i]),
        y: boostData.y.filter((_, i) => boostData.labels[i] === 1 && misclassified[i]),
        mode: 'markers',
        name: 'Class 1 (Misclassified)',
        marker: { 
            color: '#f97316',
            size: boostData.weights.filter((_, i) => boostData.labels[i] === 1 && misclassified[i]).map(w => w * 5),
            line: { color: '#ef4444', width: 2 }
        }
    };
    
    const splitLine = {
        x: split_dim === 'x' ? [split_val, split_val] : [0, 10],
        y: split_dim === 'y' ? [split_val, split_val] : [0, 10],
        mode: 'lines', 
        name: 'Decision Boundary',
        line: { color: '#ef4444', dash: 'dash' }
    };

    // Update weights for the next round
    misclassified.forEach((is_wrong, i) => {
        if(is_wrong) boostData.weights[i] *= 2;
    });

    Plotly.newPlot(boostPlotDiv, [trace0Correct, trace0Wrong, trace1Correct, trace1Wrong, splitLine], {
        title: `Weak Learner #${round + 1}`,
        xaxis: {range: [0, 10]}, 
        yaxis: {range: [0, 10]}, 
        showlegend: false,
        margin: {t:40, r:10, b:20, l:20}
    });
}


    nextBoostBtn.addEventListener('click', () => {
        boostRound++;
        if (boostRound % 3 === 0) { // Reset weights after a full cycle
            boostData.weights.fill(1);
        }
        updateBoostingPlot();
    });

    boostData = generateBoostData();
    updateBoostingPlot();
});
</script>