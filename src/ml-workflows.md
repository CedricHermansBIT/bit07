---
layout: base.njk
title: "Chapter 10: Building Robust ML Workflows"
---

<!-- Header -->
<div class="bg-gradient-to-r from-gray-50 to-slate-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 10: Building Robust ML Workflows</h2>
    <p class="text-gray-700 leading-relaxed">As models become more complex, the number of pre-processing steps (scaling, encoding, imputation) can become unwieldy. Performing these steps manually is not only tedious but is a major source of a critical error known as <strong>data leakage</strong>, which can completely invalidate your results.</p>
</div>

<!-- 1. The #1 Pitfall: Data Leakage -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">The Critical Pitfall: Data Leakage</h3>
    <p class="text-gray-700 mb-4">Data leakage occurs when information from outside the training data is used to create the model. The most common mistake is performing a pre-processing step (like scaling) on the *entire dataset* before splitting for cross-validation. This "leaks" information from the validation folds into the training process, leading to overly optimistic performance estimates.</p>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: The Effect of Data Leakage</h4>
    <p class="text-gray-600 mb-4">Compare the cross-validation score of a workflow with data leakage to a correct workflow using a Pipeline. Notice how the leaky score is artificially inflated.</p>
    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <h5 class="font-bold text-center mb-2">Run Simulated 5-Fold Cross-Validation</h5>
            <div class="flex gap-4">
                <button id="run-leaky-btn" class="btn-secondary w-1/2">Run Leaky Workflow</button>
                <button id="run-correct-btn" class="btn-primary w-1/2">Run Correct Workflow</button>
            </div>
        </div>
        <div class="text-center">
            <div class="text-sm text-gray-600">Reported CV Accuracy</div>
            <div id="leakage-score-display" class="text-4xl font-mono font-bold my-2">—</div>
            <div id="leakage-verdict" class="text-sm font-semibold"></div>
        </div>
    </div>
</div>

<!-- 2. Handling Missing Data -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">A Common Challenge: Missing Data</h3>
    <p class="text-gray-700 mb-4">Biological datasets are notoriously messy. The simplest strategy is to remove rows or columns with missing data, but this can lead to a massive loss of valuable information. A better approach is <strong>imputation</strong>: filling in missing values based on the information available.</p>
    <div class="interactive-demo">
        <div id="missing-data-grid" class="grid grid-cols-5 gap-1 mb-4 p-2 bg-gray-100 rounded"></div>
        <div class="flex flex-wrap gap-2 justify-center">
            <button data-action="drop-row" class="missing-btn btn-secondary">Drop Rows w/ Missing</button>
            <button data-action="impute-mean" class="missing-btn btn-secondary">Impute with Mean</button>
            <button data-action="reset" class="missing-btn btn-primary">Reset Data</button>
        </div>
        <p id="missing-data-info" class="text-center text-sm mt-3"></p>
    </div>
</div>

<!-- 3. The Solution: Pipelines and ColumnTransformers -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">The Gold Standard: Pipelines and ColumnTransformers</h3>
    <p class="text-gray-700 mb-4">Scikit-learn provides a powerful toolkit for creating robust, reproducible, and leak-proof workflows. The two essential components are `Pipeline` and `ColumnTransformer`.</p>

    <div class="bg-gray-50 p-4 rounded-lg border">
        <h4 class="font-semibold text-center mb-4">Visualizing a Full Workflow</h4>
        <div class="border-4 border-dashed border-gray-300 p-2 rounded-xl">
            <p class="text-center font-bold">Pipeline</p>
            <div class="bg-blue-100 p-4 rounded-lg border border-blue-300">
                <p class="text-center font-bold">ColumnTransformer ('preprocessor')</p>
                <div class="flex gap-4 mt-2">
                    <!-- Numerical Pipeline -->
                    <div class="flex-1 bg-green-100 p-2 rounded border border-green-300">
                        <p class="text-center text-sm font-semibold">Numerical Pipeline</p>
                        <div class="text-center text-xs mt-1">Input: 'age', 'expression_level'</div>
                        <div class="bg-white p-1 my-1 rounded text-center text-sm shadow">Impute Median</div>
                        <div class="bg-white p-1 my-1 rounded text-center text-sm shadow">Standard Scaler</div>
                    </div>
                    <!-- Categorical Pipeline -->
                    <div class="flex-1 bg-orange-100 p-2 rounded border border-orange-300">
                        <p class="text-center text-sm font-semibold">Categorical Pipeline</p>
                         <div class="text-center text-xs mt-1">Input: 'tumor_stage'</div>
                        <div class="bg-white p-1 my-1 rounded text-center text-sm shadow">Impute Constant</div>
                        <div class="bg-white p-1 my-1 rounded text-center text-sm shadow">OneHot Encoder</div>
                    </div>
                </div>
            </div>
            <div class="text-5xl text-center my-2 text-gray-400">↓</div>
            <div class="bg-purple-100 p-4 rounded-lg border border-purple-300 text-center">
                <p class="font-bold">RandomForestClassifier ('classifier')</p>
            </div>
        </div>
    </div>
     <div class="highlight mt-6">
        <strong>How it Prevents Leakage:</strong> When you call <code>.fit()</code> on this entire pipeline, scikit-learn is smart. During cross-validation, it will only fit the imputer and scaler on the *training portion* of each fold, and then use that learned transformation on the validation portion. This automates the correct, leak-proof procedure.
     </div>
</div>

<!-- 4. FeatureUnion -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">Advanced Technique: `FeatureUnion`</h3>
    <p class="text-gray-700 mb-4">While `ColumnTransformer` applies different steps to different columns, `FeatureUnion` applies different transformers to the *same* columns in parallel and then concatenates their outputs. This is a powerful technique for feature engineering.</p>
    <div class="bg-gray-50 p-4 rounded-lg border">
        <h4 class="font-semibold text-center mb-4">ColumnTransformer vs. FeatureUnion</h4>
        <div class="grid grid-cols-2 gap-4 text-center text-sm">
            <div>
                <strong class="text-blue-600">ColumnTransformer</strong>
                <div class="mt-2 p-2 border-2 border-blue-400 rounded-lg bg-white">
                    <p>All Gene Data</p>
                    <div class="text-2xl">↓</div>
                    <div class="flex gap-2">
                        <div class="flex-1 p-1 bg-green-100 rounded">Numerical Columns</div>
                        <div class="flex-1 p-1 bg-orange-100 rounded">Categorical Columns</div>
                    </div>
                </div>
                <p class="text-xs mt-1"><strong>Use Case:</strong> Pre-processing a dataset with mixed data types.</p>
            </div>
            <div>
                <strong class="text-purple-600">FeatureUnion</strong>
                 <div class="mt-2 p-2 border-2 border-purple-400 rounded-lg bg-white">
                    <p>All Gene Data</p>
                    <div class="text-2xl">↘ ↙</div>
                    <div class="flex gap-2">
                        <div class="flex-1 p-1 bg-purple-200 rounded">PCA Features</div>
                        <div class="flex-1 p-1 bg-purple-200 rounded">Biomarker Features</div>
                    </div>
                </div>
                <p class="text-xs mt-1"><strong>Use Case:</strong> Combining global patterns (PCA) with specific, known features (biomarkers).</p>
            </div>
        </div>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- DATA LEAKAGE DEMO ---
    const runLeakyBtn = document.getElementById('run-leaky-btn');
    const runCorrectBtn = document.getElementById('run-correct-btn');
    const scoreDisplay = document.getElementById('leakage-score-display');
    const verdictDisplay = document.getElementById('leakage-verdict');

    const runSimulation = (isLeaky) => {
        const baseScore = 0.85; // True score
        const leakBonus = 0.08; // Artificial boost from leakage
        const score = isLeaky ? baseScore + leakBonus : baseScore;
        
        scoreDisplay.textContent = `${(score + (Math.random() - 0.5) * 0.02).toFixed(3)}`;
        if (isLeaky) {
            verdictDisplay.textContent = 'Deceptively High! (Unreliable)';
            verdictDisplay.className = 'text-sm font-semibold text-red-600';
        } else {
            verdictDisplay.textContent = 'Correct and Reliable Score';
            verdictDisplay.className = 'text-sm font-semibold text-green-600';
        }
    };
    runLeakyBtn.addEventListener('click', () => runSimulation(true));
    runCorrectBtn.addEventListener('click', () => runSimulation(false));


    // --- MISSING DATA DEMO ---
    const missingGrid = document.getElementById('missing-data-grid');
    const missingInfo = document.getElementById('missing-data-info');
    const missingBtns = document.querySelectorAll('.missing-btn');
    let originalMissingData;

    const createGrid = () => {
        originalMissingData = Array.from({length: 5}, () => 
            Array.from({length: 5}, () => (Math.random() > 0.15 ? (Math.random()*10).toFixed(1) : null))
        );
        renderGrid(originalMissingData);
    };

    const renderGrid = (data) => {
        missingGrid.innerHTML = '';
        data.forEach(row => {
            row.forEach(cell => {
                const div = document.createElement('div');
                div.className = 'h-10 flex items-center justify-center rounded text-xs font-mono';
                if (cell === null) {
                    div.classList.add('bg-red-200');
                    div.textContent = 'NaN';
                } else {
                    div.classList.add('bg-blue-200');
                    div.textContent = cell;
                }
                missingGrid.appendChild(div);
            });
        });
        missingInfo.textContent = `Current dataset size: ${data.length} rows x ${data[0]?.length || 0} columns.`;
    };

    const handleMissingAction = (action) => {
        let newData;
        switch(action) {
            case 'drop-row':
                newData = originalMissingData.filter(row => !row.includes(null));
                break;
            case 'impute-mean':
                newData = JSON.parse(JSON.stringify(originalMissingData));
                for(let c=0; c<newData[0].length; c++) {
                    let sum = 0, count = 0;
                    for(let r=0; r<newData.length; r++) {
                        if (newData[r][c] !== null) {
                           sum += parseFloat(newData[r][c]);
                           count++;
                        }
                    }
                    const mean = (sum/count).toFixed(1);
                    for(let r=0; r<newData.length; r++) {
                        if (newData[r][c] === null) {
                           newData[r][c] = mean;
                        }
                    }
                }
                break;
            default: // reset
                createGrid();
                return;
        }
        renderGrid(newData);
    };

    missingBtns.forEach(btn => btn.addEventListener('click', () => handleMissingAction(btn.dataset.action)));
    createGrid();
});
</script>