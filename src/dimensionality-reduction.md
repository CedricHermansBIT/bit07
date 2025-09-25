---
layout: base.njk
title: "Chapter 9: Unsupervised Learning - Dimensionality Reduction"
---

<!-- Header -->
<div class="bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 9: Unsupervised Learning - Dimensionality Reduction</h2>
    <p class="text-gray-700 leading-relaxed">Modern bioinformatics datasets are often high-dimensional (e.g., an RNA-seq experiment can measure >20,000 genes). This presents several challenges, collectively known as the <strong>"Curse of Dimensionality,"</strong> making it hard to find patterns, visualize data, and train effective models. Dimensionality reduction is the process of transforming data into a lower-dimensional space while retaining as much meaningful information as possible.</p>
</div>

<!-- 1. Principal Component Analysis (PCA) -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">Linear Reduction: Principal Component Analysis (PCA)</h3>
    <p class="text-gray-700 mb-4">PCA is the most widely used linear dimensionality reduction technique. It transforms the original features into a new set of uncorrelated features called <strong>principal components (PCs)</strong>. The first PC (PC1) is the direction that captures the largest possible variance in the data, PC2 captures the second-largest while being orthogonal to PC1, and so on.</p>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: How PCA Works</h4>
    <p class="text-gray-600 mb-4">Step through the PCA process to see how it finds the new axes that best represent this patient dataset.</p>

    <div class="interactive-demo grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Plot Area -->
        <div>
             <div id="pca-plot" style="width:100%; height:300px;"></div>
        </div>
        <!-- Controls and Explanation -->
        <div class="space-y-3 self-center">
            <div id="pca-status" class="text-center font-semibold p-3 bg-white rounded-lg">Original patient data with two correlated features.</div>
            <button id="pca-step-btn" class="btn-primary w-full">Step 1: Standardize Data</button>
            <button id="pca-reset-btn" class="btn-secondary w-full">Reset Demo</button>
            <div class="mt-4">
                 <h5 class="font-bold text-center">Explained Variance Ratio</h5>
                 <div id="variance-plot" style="width:100%; height:150px;"></div>
            </div>
        </div>
    </div>
</div>

<!-- 2. Non-Linear Methods -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">Non-Linear Methods: t-SNE and UMAP</h3>
    <p class="text-gray-700 mb-4">While PCA is excellent for capturing linear relationships, many biological datasets have complex, non-linear structures. For these cases, manifold learning algorithms like <strong>t-SNE</strong> and <strong>UMAP</strong> are indispensable, especially for visualizing single-cell RNA-sequencing data.</p>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Visual Comparison: PCA vs. t-SNE vs. UMAP</h4>
    <p class="text-gray-600 mb-4">This dataset contains several distinct cell types. Click the buttons to see how well each algorithm separates these clusters in a 2D visualization. Note that dimensionality reduction techniques do not perform clustering, the clusters you see are either labels from the data, or produced by a clustering algorithm.</p>
    
    <div class="interactive-demo">
        <div class="flex justify-center gap-4 mb-4">
            <button data-method="pca" class="vis-btn btn-primary">PCA</button>
            <button data-method="tsne" class="vis-btn btn-secondary">t-SNE</button>
            <button data-method="umap" class="vis-btn btn-secondary">UMAP</button>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div id="vis-plot-container" class="md:col-span-2 flex justify-center items-center">
                <img id="vis-plot-img" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*qT2mgOCuUZOKqtwwDKnuwA.png" alt="PCA Plot" class="max-w-full md:max-w-lg rounded-lg border">
            </div>
            <div id="vis-explanation" class="md:col-span-2 text-center p-3 bg-white rounded-lg">
                <!-- JS will populate this -->
            </div>
        </div>
    </div>
</div>


<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- PCA STEP-BY-STEP DEMO ---
    const pcaPlotDiv = document.getElementById('pca-plot');
    const pcaStepBtn = document.getElementById('pca-step-btn');
    const pcaResetBtn = document.getElementById('pca-reset-btn');
    const pcaStatus = document.getElementById('pca-status');
    const variancePlotDiv = document.getElementById('variance-plot');

    let pcaState = 0;
    let pcaData, standardizedData, principalComponents;

    const generatePCAData = () => {
        const data = { x: [], y: [] };
        const mean = [5, 5];
        const cov = [[1, 0.8], [0.8, 1]]; // Correlated data
        for (let i = 0; i < 100; i++) {
            const z1 = Math.random() * 2 - 1;
            const z2 = Math.random() * 2 - 1;
            data.x.push(mean[0] + cov[0][0] * z1 + cov[0][1] * z2);
            data.y.push(mean[1] + cov[1][0] * z1 + cov[1][1] * z2);
        }
        return data;
    };

    const standardize = (data) => {
        const meanX = data.x.reduce((a, b) => a + b) / data.x.length;
        const meanY = data.y.reduce((a, b) => a + b) / data.y.length;
        const stdX = Math.sqrt(data.x.map(v => (v - meanX)**2).reduce((a, b) => a + b) / data.x.length);
        const stdY = Math.sqrt(data.y.map(v => (v - meanY)**2).reduce((a, b) => a + b) / data.y.length);
        return {
            x: data.x.map(v => (v - meanX) / stdX),
            y: data.y.map(v => (v - meanY) / stdY)
        };
    };

    const findPCs = (data) => {
        // Calculate covariance matrix entries
        const covXX = data.x.map(v => v*v).reduce((a,b)=>a+b) / data.x.length;
        const covYY = data.y.map(v => v*v).reduce((a,b)=>a+b) / data.x.length;
        const covXY = data.x.map((v, i) => v * data.y[i]).reduce((a,b)=>a+b) / data.x.length;
        
        // Calculate eigenvalues
        const trace = covXX + covYY;
        const det = covXX * covYY - covXY * covXY;
        const eig1 = trace / 2 + Math.sqrt(trace**2 / 4 - det);
        const eig2 = trace / 2 - Math.sqrt(trace**2 / 4 - det);
        
        // Calculate first eigenvector (PC1)
        let vec1;
        if (Math.abs(covXY) > 1e-10) {
            vec1 = {
                x: 1,
                y: (eig1 - covXX) / covXY
            };
        } else {
            vec1 = {
                x: covXX >= covYY ? 1 : 0,
                y: covXX >= covYY ? 0 : 1
            };
        }
        
        // Calculate second eigenvector (PC2) as perpendicular to PC1
        const vec2 = {
            x: -vec1.y,
            y: vec1.x
        };
        
        // Normalize the vectors
        const norm1 = Math.sqrt(vec1.x**2 + vec1.y**2);
        const norm2 = Math.sqrt(vec2.x**2 + vec2.y**2);
        
        return {
            pc1: { x: vec1.x / norm1, y: vec1.y / norm1, var: eig1 },
            pc2: { x: vec2.x / norm2, y: vec2.y / norm2, var: eig2 }
        };
    };


    const plotPCA = () => {
        const dataToPlot = (pcaState > 0) ? standardizedData : pcaData;
        const traces = [{
            x: dataToPlot.x, y: dataToPlot.y, mode: 'markers', type: 'scatter',
            marker: { color: '#60a5fa', size: 8 }
        }];

        if (pcaState >= 2) {
            traces.push({ x: [0, principalComponents.pc1.x * 2], y: [0, principalComponents.pc1.y * 2], mode: 'lines', line: { color: 'red', width: 3 }, name: 'PC1' });
            traces.push({ x: [0, principalComponents.pc2.x * 2], y: [0, principalComponents.pc2.y * 2], mode: 'lines', line: { color: 'green', width: 3 }, name: 'PC2' });
        }
        
        if (pcaState === 3) {
            const projectedX = standardizedData.x.map((v, i) => v * principalComponents.pc1.x + standardizedData.y[i] * principalComponents.pc1.y);
            traces[0].marker.color = '#d1d5db';
            traces.push({
                x: projectedX.map(p => p * principalComponents.pc1.x),
                y: projectedX.map(p => p * principalComponents.pc1.y),
                mode: 'markers', type: 'scatter', marker: { color: 'red', size: 8 }
            });
        }
        
        Plotly.newPlot(pcaPlotDiv, traces, {
            title: 'Patient Data',
            xaxis: { title: 'Feature 1', range: pcaState > 0 ? [-3, 3] : [0, 10], constrain: 'domain' },
            yaxis: { title: 'Feature 2', range: pcaState > 0 ? [-3, 3] : [0, 10], scaleanchor: 'x', scaleratio: 1 },
            // Enforce equal aspect ratio so PC2 is visually perpendicular to PC1
            showlegend: false, margin: {t:40, r:10, b:40, l:40}
        });

        const totalVar = (principalComponents?.pc1.var || 0) + (principalComponents?.pc2.var || 0);
        const var1 = (principalComponents?.pc1.var || 0) / (totalVar || 1);
        const var2 = (principalComponents?.pc2.var || 0) / (totalVar || 1);
        Plotly.newPlot(variancePlotDiv, [{
            x: ['PC1', 'PC2'], y: [var1, var2], type: 'bar', marker: {color: ['red', 'green']}
        }], {margin: {t:10, r:10, b:20, l:40}, yaxis: {range:[0,1], title: 'Explained Var.'}});

    };

    const runPCAStep = () => {
        pcaState++;
        if (pcaState === 1) { // Standardize
            standardizedData = standardize(pcaData);
            pcaStatus.textContent = "Data is centered and scaled to have zero mean and unit variance.";
            pcaStepBtn.textContent = "Step 2: Find Principal Components";
        } else if (pcaState === 2) { // Find PCs
            principalComponents = findPCs(standardizedData);
            pcaStatus.textContent = "PC1 (red) is the axis of max variance. PC2 (green) is orthogonal to PC1.";
            pcaStepBtn.textContent = "Step 3: Project Data onto PC1";
        } else if (pcaState === 3) { // Project
            pcaStatus.textContent = "Data is projected onto PC1. The dimension is now reduced from 2D to 1D.";
            pcaStepBtn.disabled = true;
        }
        plotPCA();
    };

    const resetPCA = () => {
        pcaState = 0;
        pcaData = generatePCAData();
        standardizedData = null;
        principalComponents = null;
        pcaStatus.textContent = "Original patient data with two correlated features.";
        pcaStepBtn.textContent = "Step 1: Standardize Data";
        pcaStepBtn.disabled = false;
        plotPCA();
    };

    pcaStepBtn.addEventListener('click', runPCAStep);
    pcaResetBtn.addEventListener('click', resetPCA);
    resetPCA();

    // --- VISUAL COMPARISON DEMO ---
    const visBtns = document.querySelectorAll('.vis-btn');
    const visImg = document.getElementById('vis-plot-img');
    const visExplanation = document.getElementById('vis-explanation');
    const visContent = {
        pca: {
            img: "https://miro.medium.com/v2/resize:fit:640/format:webp/1*qT2mgOCuUZOKqtwwDKnuwA.png",
            text: "<strong>PCA</strong> captures the main global variance but struggles to separate the non-linear, intertwined clusters. It shows a 'cloud' with some separation but misses the fine-grained local structure."
        },
        tsne: {
            img: "https://miro.medium.com/v2/resize:fit:640/format:webp/1*3cU0mnZWFmJjP6_rSznObA.png",
            text: "<strong>t-SNE</strong> excels at revealing local neighborhood structures. It creates well-separated, dense clusters, making it excellent for visualization. However, the distances between clusters are not meaningful."
        },
        umap: {
            img: "https://miro.medium.com/v2/resize:fit:720/format:webp/1*O4W7gLYZGfkDsAla7-er2g.png",
            text: "<strong>UMAP</strong> is often faster than t-SNE and is better at preserving the global data structure. Notice how it separates the clusters well, but also maintains some of their relative positioning, unlike t-SNE."
        }
    };
    
    visBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const method = e.target.dataset.method;
            visBtns.forEach(b => {
                b.classList.remove('btn-primary');
                b.classList.add('btn-secondary');
            });
            e.target.classList.add('btn-primary');
            e.target.classList.remove('btn-secondary');

            visImg.src = visContent[method].img;
            visExplanation.innerHTML = visContent[method].text;
        });
    });
    // Set initial explanation
    visExplanation.innerHTML = visContent.pca.text;
});
</script>