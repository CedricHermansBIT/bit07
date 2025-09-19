---
layout: base.njk
title: "Chapter 5: Working with Imbalanced Data"
---

<!-- Header -->
<div class="bg-gradient-to-r from-red-50 to-yellow-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 5: Working with Imbalanced Data</h2>
    <p class="text-gray-700 leading-relaxed">In many real-world classification problems, the distribution of classes is not equal. Imbalanced data is the norm, not the exception, in bioinformatics, where disease states or pathogenic variants are often rare. This presents a significant challenge for training effective machine learning models.</p>
</div>

<!-- 1. The Core Problem: The Accuracy Paradox -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">The Core Problem: The Accuracy Paradox</h3>
    <p class="text-gray-700 mb-4">When a dataset is imbalanced, the standard evaluation metric, **accuracy**, becomes deeply misleading. A naive model that simply predicts the majority class for every single sample can achieve a very high accuracy score, despite being completely useless for identifying the minority class (which is often the class of interest).</p>

    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <label class="block text-sm font-medium"># of Minority Samples (e.g., Cancer Patients)</label>
            <input id="minority-slider" type="range" min="1" max="100" step="1" value="10" class="parameter-slider">
            <p class="text-sm text-center"><span id="minority-count">10</span> Cancer Patients vs. <span id="majority-count">990</span> Healthy Individuals</p>
            <p class="text-sm text-gray-600 mt-4">Below is the performance of a "naive" model that always predicts "Healthy". Notice how high the accuracy is.</p>
        </div>
        <div class="text-center">
            <div class="text-lg font-bold text-red-600">Naive Model Accuracy</div>
            <div id="accuracy-paradox-display" class="text-5xl font-mono font-bold text-red-600 my-2">99.0%</div>
            <div id="paradox-cm" class="grid grid-cols-3 gap-1 text-sm font-semibold w-full max-w-xs mx-auto"></div>
        </div>
    </div>
    <div class="alert-warning mt-4"><strong>Conclusion:</strong> Relying on accuracy for imbalanced data is dangerous. We must use more informative metrics and specialized techniques.</div>
</div>

<!-- 2. Strategies for Handling Imbalance -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">A Toolkit for Handling Imbalanced Data</h3>
    <p class="text-gray-700 mb-6">Several strategies can be employed to mitigate the effects of class imbalance. These can be broadly categorized into data-level, algorithm-level, and evaluation-level approaches.</p>

    <!-- Data-Level Solutions -->
    <h4 class="text-xl font-semibold text-gray-700 mb-4">1. Data-Level Solutions: Resampling Techniques</h4>
    <p class="text-gray-600 mb-4">Resampling techniques modify the training dataset to create a more balanced class distribution. Visualize their effects on the data below.</p>
    <div class="interactive-demo grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        <div class="space-y-3">
            <h5 class="font-bold text-center">Select a Technique</h5>
            <button data-technique="original" class="resample-btn btn-primary w-full">Original Data</button>
            <button data-technique="under" class="resample-btn btn-secondary w-full">Random Undersampling</button>
            <button data-technique="over" class="resample-btn btn-secondary w-full">Random Oversampling</button>
            <button data-technique="smote" class="resample-btn btn-secondary w-full">SMOTE</button>
            <div class="p-3 bg-gray-100 rounded text-center">
                <p class="text-sm font-semibold">Class Counts</p>
                <p id="class-counts-display" class="text-sm font-mono"></p>
            </div>
        </div>
        <div class="md:col-span-2">
            <div id="resampling-plot" style="width:100%;height:300px;"></div>
        </div>
    </div>

    <!-- Algorithm-Level Solutions -->
    <h4 class="text-xl font-semibold text-gray-700 mt-8 mb-4">2. Algorithm-Level Solutions: Class Weighting</h4>
    <p class="text-gray-600 mb-4">This is often a simple and effective first step. Many algorithms have a <code>class_weight</code> parameter that modifies the loss function to penalize misclassifications of the minority class more heavily. It doesn't require modifying the data itself.</p>
    <div class="code-block">
<pre><code>from sklearn.svm import SVC

# Let the algorithm automatically adjust weights to be
# inversely proportional to class frequencies.
model = SVC(class_weight='balanced')

# Or, manually assign a higher weight to the minority class (e.g., class 1)
# This tells the model a mistake on class 1 is 10 times worse.
model = SVC(class_weight={0: 1, 1: 10})</code></pre>
    </div>

    <!-- Evaluation-Level Solutions -->
    <h4 class="text-xl font-semibold text-gray-700 mt-8 mb-4">3. Evaluation-Level Solutions: Choosing the Right Scorer</h4>
    <p class="text-gray-600 mb-4">As we've seen, accuracy is a poor metric. Instead, we should use metrics derived from the confusion matrix that provide a better picture of performance on the minority class, such as **Precision, Recall, and F1-Score**.</p>
    <div class="highlight">
        <strong>Making it Actionable: Hyperparameter Tuning</strong>
        <p>When performing hyperparameter tuning with <code>GridSearchCV</code> or <code>RandomizedSearchCV</code>, it is crucial to change the default scoring metric from 'accuracy' to one that is robust to imbalance.</p>
<pre><code>grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    # Optimize for F1-score instead of accuracy!
    scoring='f1_weighted' // Other options: 'roc_auc', 'balanced_accuracy'
)</code></pre>
    </div>
</div>

<!-- Practical Implementation -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">Practical Implementation with `imbalanced-learn`</h3>
    <p class="text-gray-700 mb-4">The <code>imbalanced-learn</code> library for Python provides easy-to-use implementations of these techniques, designed to work seamlessly with scikit-learn.</p>
    <div class="code-block">
<pre><code class="language-python"># First, install the library:
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
# Create a SMOTE object and fit it to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Now you can fit the classifier as usual
model.fit(X_train_resampled, y_train_resampled)
# Test the model on the test set
predictions = model.predict(X_test)
# ... Perform evaluation metrics</code></pre>
    </div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- 1. ACCURACY PARADOX DEMO ---
    const minoritySlider = document.getElementById('minority-slider');
    const minorityCountSpan = document.getElementById('minority-count');
    const majorityCountSpan = document.getElementById('majority-count');
    const accuracyDisplay = document.getElementById('accuracy-paradox-display');
    const paradoxCmDiv = document.getElementById('paradox-cm');

    function updateAccuracyParadox() {
        const minorityCount = parseInt(minoritySlider.value);
        const majorityCount = 1000 - minorityCount;
        
        minorityCountSpan.textContent = minorityCount;
        majorityCountSpan.textContent = majorityCount;
        
        // Naive model always predicts majority class (0)
        const tn = majorityCount;
        const fp = 0;
        const fn = minorityCount;
        const tp = 0;
        
        const accuracy = (tn / (tn + fn)) * 100;
        accuracyDisplay.textContent = `${accuracy.toFixed(1)}%`;
        
        paradoxCmDiv.innerHTML = `
            <div></div><div class="bg-gray-200 p-1">Predicted Healthy</div><div class="bg-gray-200 p-1">Predicted Cancer</div>
            <div class="bg-gray-200 p-1">Actual Cancer</div><div class="bg-red-200 p-2">FN: ${fn}</div><div class="bg-green-200 p-2">TP: ${tp}</div>
            <div class="bg-gray-200 p-1">Actual Healthy</div><div class="bg-blue-200 p-2">TN: ${tn}</div><div class="bg-yellow-200 p-2">FP: ${fp}</div>
        `;
    }
    minoritySlider.addEventListener('input', updateAccuracyParadox);
    updateAccuracyParadox();

    // --- 2. RESAMPLING DEMO ---
    const resampleBtns = document.querySelectorAll('.resample-btn');
    const resamplingPlotDiv = document.getElementById('resampling-plot');
    const classCountsDisplay = document.getElementById('class-counts-display');

    const generateImbalancedData = () => {
        let data = { x: [], y: [], labels: [] };
        // Majority class (0)
        for(let i=0; i<200; i++) {
            data.x.push(Math.random() * 4 + 1);
            data.y.push(Math.random() * 4 + 1);
            data.labels.push(0);
        }
        // Minority class (1)
        for(let i=0; i<20; i++) {
            data.x.push(Math.random() * 2 + 4);
            data.y.push(Math.random() * 2 + 4);
            data.labels.push(1);
        }
        return data;
    };

    const originalData = generateImbalancedData();

    const plotData = (data, title) => {
        const trace0 = {
            x: data.x.filter((_, i) => data.labels[i] === 0),
            y: data.y.filter((_, i) => data.labels[i] === 0),
            mode: 'markers', name: 'Majority (Healthy)', type: 'scatter',
            marker: { color: '#60a5fa', size: 7, opacity: 0.7 }
        };
        const trace1 = {
            x: data.x.filter((_, i) => data.labels[i] === 1),
            y: data.y.filter((_, i) => data.labels[i] === 1),
            mode: 'markers', name: 'Minority (Disease)', type: 'scatter',
            marker: { color: '#f97316', size: 7 }
        };
        const layout = {
            title: title,
            xaxis: { showticklabels: false, zeroline: false },
            yaxis: { showticklabels: false, zeroline: false },
            margin: { t: 40, r: 10, b: 10, l: 10 },
            legend: { x: 0, y: 1 }
        };
        Plotly.newPlot(resamplingPlotDiv, [trace0, trace1], layout);
        const majorityCount = data.labels.filter(l => l === 0).length;
        const minorityCount = data.labels.filter(l => l === 1).length;
        classCountsDisplay.innerHTML = `Majority: ${majorityCount}<br>Minority: ${minorityCount}`;
    };

    const applyResampling = (technique) => {
        let newData;
        let title = '';
        const minorityPoints = originalData.x.map((_, i) => ({x: originalData.x[i], y: originalData.y[i], label: originalData.labels[i]})).filter(p => p.label === 1);
        const majorityPoints = originalData.x.map((_, i) => ({x: originalData.x[i], y: originalData.y[i], label: originalData.labels[i]})).filter(p => p.label === 0);

        switch(technique) {
            case 'under':
                title = 'Random Undersampling';
                const shuffledMajority = majorityPoints.sort(() => 0.5 - Math.random());
                const underSampledMajority = shuffledMajority.slice(0, minorityPoints.length);
                const combinedUnder = [...minorityPoints, ...underSampledMajority];
                newData = {
                    x: combinedUnder.map(p => p.x),
                    y: combinedUnder.map(p => p.y),
                    labels: combinedUnder.map(p => p.label)
                };
                break;
            case 'over':
                title = 'Random Oversampling';
                let oversampledMinority = [...minorityPoints];
                while(oversampledMinority.length < majorityPoints.length) {
                    // add tiny amount of offset
                    let p = minorityPoints[Math.floor(Math.random() * minorityPoints.length)];
                    const offsetX = (Math.random() - 0.5) * 0.05;
                    const offsetY = (Math.random() - 0.5) * 0.05;
                    oversampledMinority.push({x: p.x + offsetX, y: p.y + offsetY, label: 1});
                }
                const combinedOver = [...majorityPoints, ...oversampledMinority];
                 newData = {
                    x: combinedOver.map(p => p.x),
                    y: combinedOver.map(p => p.y),
                    labels: combinedOver.map(p => p.label)
                };
                break;
            case 'smote':
                title = 'SMOTE (Synthetic Samples)';
                let smoteMinority = [...minorityPoints];
                while(smoteMinority.length < majorityPoints.length) {
                    const p1 = minorityPoints[Math.floor(Math.random() * minorityPoints.length)];
                    const p2 = minorityPoints[Math.floor(Math.random() * minorityPoints.length)];
                    const gap = Math.random();
                    const newX = p1.x + (p2.x - p1.x) * gap;
                    const newY = p1.y + (p2.y - p1.y) * gap;
                    smoteMinority.push({x: newX, y: newY, label: 1});
                }
                 const combinedSmote = [...majorityPoints, ...smoteMinority];
                 newData = {
                    x: combinedSmote.map(p => p.x),
                    y: combinedSmote.map(p => p.y),
                    labels: combinedSmote.map(p => p.label)
                };
                break;
            default:
                title = 'Original Imbalanced Data';
                newData = originalData;
        }
        plotData(newData, title);
    };

    resampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            resampleBtns.forEach(b => {
                b.classList.remove('btn-primary');
                b.classList.add('btn-secondary');
            });
            e.target.classList.remove('btn-secondary');
            e.target.classList.add('btn-primary');
            applyResampling(e.target.dataset.technique);
        });
    });
    
    applyResampling('original');
});
</script>

<script>
document.addEventListener('DOMContentLoaded', () => {
    // run code once
    applyResampling('original');
})
</script>