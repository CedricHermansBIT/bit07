---
layout: base.njk
title: "Chapter 9: Unsupervised Learning - Clustering"
---

<!-- Header -->
<div class="bg-gradient-to-r from-sky-50 to-blue-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 9: Unsupervised Learning - Clustering</h2>
    <p class="text-gray-700 leading-relaxed">In all previous chapters, we used labeled data to train our models. But what if we have no labels? <strong>Unsupervised Learning</strong> provides tools to extract meaningful insights from unlabeled data. This chapter focuses on its most fundamental task: <strong>Clustering</strong>, the process of finding natural groups in data.</p>
</div>

<!-- 1. What is Clustering? -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">What is Clustering and Why is it Useful?</h3>
    <p class="text-gray-700 mb-4">Clustering is the task of partitioning a dataset into groups, or "clusters," such that data points within the same cluster are more similar to each other than to those in other clusters. It's a cornerstone of exploratory data analysis in bioinformatics.</p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div class="bg-blue-50 p-3 rounded-lg">
            <h4 class="font-semibold text-blue-800">Gene Co-expression Analysis</h4>
            <p>Group genes with similar expression patterns across conditions to find functionally related or co-regulated genes.</p>
        </div>
        <div class="bg-green-50 p-3 rounded-lg">
            <h4 class="font-semibold text-green-800">Patient Stratification</h4>
            <p>Group patients based on their molecular profiles (e.g., genomics, proteomics) to identify novel disease subtypes for personalized treatment.</p>
        </div>
    </div>
</div>

<!-- 2. K-Means Clustering -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">Partitioning Clustering: K-Means</h3>
    <p class="text-gray-700 mb-4">K-Means is the most popular partitioning algorithm. It aims to partition data into a pre-defined number of clusters (`k`), where each point belongs to the cluster with the nearest mean (the "centroid").</p>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: The K-Means Algorithm Step-by-Step</h4>
    <p class="text-gray-600 mb-4">You are the algorithm! Step through the process to see how K-Means finds its clusters. The goal is to group the three "gene modules" in this synthetic expression data.</p>
    <p class="text-gray-600 mb-4">Note that if you try long enough, you might encounter scenarios that are not optimal, and the model gets stuck. That is why we will often try multiple different starting centroids.</p>
    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
            <div id="kmeans-plot" style="width:100%; height:300px;"></div>
        </div>
        <div class="space-y-3 self-center">
            <div id="kmeans-status" class="text-center font-semibold p-3 bg-white rounded-lg">Press "Initialize" to begin.</div>
            <button id="kmeans-step-btn" class="btn-primary w-full">Initialize Centroids</button>
            <button id="kmeans-reset-btn" class="btn-secondary w-full">Reset Demo</button>
        </div>
    </div>

    <h4 class="text-xl font-bold text-gray-800 mt-8 mb-4">The Crucial Question: How to Choose `k`?</h4>
    <p class="text-gray-600 mb-4">The main challenge of K-Means is that you must specify the number of clusters, `k`, in advance. Two common methods help guide this choice:</p>
    <div class="interactive-demo grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
        <div>
            <label class="block text-sm font-medium">Number of Clusters (k)</label>
            <input id="k-slider" type="range" min="1" max="10" step="1" value="3" class="parameter-slider">
            <div id="k-value-display" class="text-center font-semibold mt-2">k = 3</div>
            <div class="mt-4 grid grid-cols-2 gap-2 text-center text-sm">
                <div class="bg-white p-2 rounded-lg">
                    <div class="font-bold">SSE (Inertia)</div>
                    <div id="sse-value" class="font-mono">--</div>
                </div>
                <div class="bg-white p-2 rounded-lg">
                    <div class="font-bold">Silhouette Score</div>
                    <div id="silhouette-value" class="font-mono">--</div>
                </div>
            </div>
        </div>
        <div>
            <div id="elbow-plot" style="width:100%; height:150px;"></div>
            <div id="silhouette-plot" style="width:100%; height:150px;"></div>
        </div>
    </div>
</div>

<!-- 3. Hierarchical Clustering -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">Hierarchical Clustering</h3>
    <p class="text-gray-700 mb-4">Unlike K-Means, hierarchical clustering builds a hierarchy of nested clusters and does not require you to specify the number of clusters beforehand. The result is visualized as a tree-like diagram called a <strong>dendrogram</strong>.</p>
    <div class="bg-gray-50 p-4 rounded-lg">
        <h4 class="font-semibold text-center mb-2">Interpreting a Dendrogram</h4>
        <img src="https://towardsdatascience.com/wp-content/uploads/2021/05/1VvOVxdBb74IOxxF2RmthCQ.png" alt="Dendrogram Diagram" class="w-full max-w-lg mx-auto rounded-lg border">
        <p class="text-xs text-center mt-2">The y-axis represents the distance between clusters. You can "cut" the tree at a certain height to get a specific number of clusters. The horizontal red line here "cuts" the tree to produce 4 distinct clusters.</p>
    </div>
     <div class="highlight mt-4">
        <strong>Agglomerative (Bottom-Up) vs. Divisive (Top-Down):</strong> The most common strategy is <strong>agglomerative</strong>, which starts with each data point in its own cluster and iteratively merges the two closest clusters until only one remains.
     </div>
</div>


<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/simple-statistics@7.7.0/dist/simple-statistics.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- K-MEANS STEP-BY-STEP DEMO ---
    const kmeansPlotDiv = document.getElementById('kmeans-plot');
    const kmeansStepBtn = document.getElementById('kmeans-step-btn');
    const kmeansResetBtn = document.getElementById('kmeans-reset-btn');
    const kmeansStatus = document.getElementById('kmeans-status');
    let kmeansState = 0; // 0: uninitialized, 1: initialized, 2: assigned, 3: updated, ...
    let kmeansData, centroids;

    const generateKmeansData = () => {
        const data = { x: [], y: [], labels: [] };
        const means = [[2, 8], [8, 8], [5, 2]];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 50; j++) {
                data.x.push(means[i][0] + (Math.random() - 0.5) * 2.5);
                data.y.push(means[i][1] + (Math.random() - 0.5) * 2.5);
            }
        }
        return data;
    };

    const dist = (p1, p2) => Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2);

    const plotKmeans = (assignments = null) => {
        const traces = [];
        const colors = ['#60a5fa', '#34d399', '#f97316'];
        
        const dataTrace = {
            x: kmeansData.x, y: kmeansData.y, mode: 'markers', type: 'scatter',
            marker: { 
                color: assignments ? assignments.map(a => colors[a]) : '#d1d5db',
                size: 8, opacity: 0.8 
            }
        };
        traces.push(dataTrace);

        if (centroids) {
            const centroidTrace = {
                x: centroids.map(c => c.x), y: centroids.map(c => c.y),
                mode: 'markers', type: 'scatter',
                marker: { color: colors, size: 15, symbol: 'cross', line: { width: 3 } }
            };
            traces.push(centroidTrace);
        }

        Plotly.newPlot(kmeansPlotDiv, traces, {
            title: 'K-Means Clustering of Gene Modules',
            xaxis: { range: [0, 10] }, yaxis: { range: [0, 10] }, showlegend: false,
            margin: { t: 30, r: 10, b: 20, l: 20 }
        });
    };

    const runKmeansStep = () => {
        if (kmeansState === 0) { // Initialize
            centroids = Array.from({length: 3}, () => ({ x: Math.random() * 10, y: Math.random() * 10 }));
            plotKmeans();
            kmeansStatus.textContent = 'Random centroids initialized.';
            kmeansStepBtn.textContent = 'Step 2: Assign Points';
            kmeansState = 1;
        } else if (kmeansState % 2 !== 0) { // Assign
            const assignments = kmeansData.x.map((x, i) => {
                let bestCentroid = 0, minDist = Infinity;
                const point = {x, y: kmeansData.y[i]};
                centroids.forEach((c, c_idx) => {
                    const d = dist(point, c);
                    if (d < minDist) {
                        minDist = d;
                        bestCentroid = c_idx;
                    }
                });
                return bestCentroid;
            });
            plotKmeans(assignments);
            kmeansStatus.textContent = 'Points assigned to nearest centroid.';
            kmeansStepBtn.textContent = 'Step 3: Update Centroids';
            kmeansState++;
        } else { // Update
            const assignments = kmeansData.x.map((x, i) => {
                let bestCentroid = 0, minDist = Infinity;
                const point = {x, y: kmeansData.y[i]};
                centroids.forEach((c, c_idx) => {
                    const d = dist(point, c);
                    if (d < minDist) {
                        minDist = d;
                        bestCentroid = c_idx;
                    }
                });
                return bestCentroid;
            });
            const newCentroids = Array.from({length: 3}, () => ({x: 0, y: 0, count: 0}));
            assignments.forEach((a, i) => {
                newCentroids[a].x += kmeansData.x[i];
                newCentroids[a].y += kmeansData.y[i];
                newCentroids[a].count++;
            });
            centroids = newCentroids.map(c => ({ x: c.x / c.count, y: c.y / c.count }));
            plotKmeans(assignments);
            kmeansStatus.textContent = 'Centroids moved to the mean of their clusters.';
            kmeansStepBtn.textContent = 'Step 2: Re-assign Points';
            kmeansState++;
        }
    };
    
    const resetKmeans = () => {
        kmeansState = 0;
        centroids = null;
        kmeansData = generateKmeansData();
        plotKmeans();
        kmeansStatus.textContent = 'Press "Initialize" to begin.';
        kmeansStepBtn.textContent = 'Step 1: Initialize Centroids';
    };

    kmeansStepBtn.addEventListener('click', runKmeansStep);
    kmeansResetBtn.addEventListener('click', resetKmeans);
    resetKmeans();

// --- CHOOSE K DEMO ---
const kSlider = document.getElementById('k-slider');
const kValueDisplay = document.getElementById('k-value-display');
const sseValue = document.getElementById('sse-value');
const silhouetteValue = document.getElementById('silhouette-value');
const elbowPlotDiv = document.getElementById('elbow-plot');
const silhouettePlotDiv = document.getElementById('silhouette-plot');

// Simple implementation of k-means
function runKMeans(data, k, maxIterations = 10) {
    // Initialize centroids randomly
    let centroids = Array.from({length: k}, () => {
        const randomIdx = Math.floor(Math.random() * data.length);
        return [data[randomIdx][0], data[randomIdx][1]];
    });
    
    let assignments = Array(data.length).fill(0);
    let iterations = 0;
    let changed = true;
    
    while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;
        
        // Assign points to nearest centroid
        for (let i = 0; i < data.length; i++) {
            let minDist = Infinity;
            let minIdx = 0;
            
            for (let j = 0; j < centroids.length; j++) {
                const dist = Math.sqrt(
                    Math.pow(data[i][0] - centroids[j][0], 2) + 
                    Math.pow(data[i][1] - centroids[j][1], 2)
                );
                
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = j;
                }
            }
            
            if (assignments[i] !== minIdx) {
                changed = true;
                assignments[i] = minIdx;
            }
        }
        
        // Update centroids
        const clusterSums = Array(k).fill().map(() => [0, 0, 0]); // [sumX, sumY, count]
        
        for (let i = 0; i < data.length; i++) {
            const centroidIdx = assignments[i];
            clusterSums[centroidIdx][0] += data[i][0];
            clusterSums[centroidIdx][1] += data[i][1];
            clusterSums[centroidIdx][2]++;
        }
        
        for (let i = 0; i < k; i++) {
            if (clusterSums[i][2] > 0) {
                centroids[i] = [
                    clusterSums[i][0] / clusterSums[i][2],
                    clusterSums[i][1] / clusterSums[i][2]
                ];
            }
        }
    }
    
    // Calculate SSE
    let sse = 0;
    for (let i = 0; i < data.length; i++) {
        const centroidIdx = assignments[i];
        sse += Math.pow(data[i][0] - centroids[centroidIdx][0], 2) + 
               Math.pow(data[i][1] - centroids[centroidIdx][1], 2);
    }
    
    // Simplified silhouette calculation
    let silhouette = 0;
    if (k > 1) {
        for (let i = 0; i < data.length; i++) {
            // Calculate average distance to points in same cluster (a)
            let intraClusterDist = 0, intraCount = 0;
            for (let j = 0; j < data.length; j++) {
                if (i !== j && assignments[i] === assignments[j]) {
                    intraClusterDist += Math.sqrt(
                        Math.pow(data[i][0] - data[j][0], 2) + 
                        Math.pow(data[i][1] - data[j][1], 2)
                    );
                    intraCount++;
                }
            }
            const a = intraCount > 0 ? intraClusterDist / intraCount : 0;
            
            // Calculate minimum average distance to points in other clusters (b)
            let minInterClusterDist = Infinity;
            for (let c = 0; c < k; c++) {
                if (c === assignments[i]) continue;
                
                let interClusterDist = 0, interCount = 0;
                for (let j = 0; j < data.length; j++) {
                    if (assignments[j] === c) {
                        interClusterDist += Math.sqrt(
                            Math.pow(data[i][0] - data[j][0], 2) + 
                            Math.pow(data[i][1] - data[j][1], 2)
                        );
                        interCount++;
                    }
                }
                
                if (interCount > 0) {
                    minInterClusterDist = Math.min(minInterClusterDist, interClusterDist / interCount);
                }
            }
            const b = minInterClusterDist === Infinity ? 0 : minInterClusterDist;
            
            // Calculate silhouette
            if (Math.max(a, b) > 0) {
                silhouette += (b - a) / Math.max(a, b);
            }
        }
        silhouette /= data.length;
    }
    
    return { 
        assignments, 
        centroids,
        sse,
        silhouette
    };
}

// Pre-calculate statistics with multiple runs per K
const chooseKData = generateKmeansData();
const points = chooseKData.x.map((x, i) => [x, chooseKData.y[i]]);
const stats = Array.from({length: 10}, (_, k_idx) => {
    const k = k_idx + 1;
    let bestResult = { sse: Infinity, silhouette: -1 };
    
    // Run 5 times with different random initializations
    for (let run = 0; run < 5; run++) {
        const result = runKMeans(points, k, 15); // Increased max iterations
        if (result.sse < bestResult.sse) {
            bestResult = result;
        }
    }
    
    return { 
        sse: bestResult.sse, 
        silhouette: bestResult.silhouette 
    };
});

function updateChooseKPlots() {
    const k = parseInt(kSlider.value);
    kValueDisplay.textContent = `k = ${k}`;
    sseValue.textContent = stats[k-1].sse.toFixed(1);
    silhouetteValue.textContent = stats[k-1].silhouette.toFixed(3);

    const elbowTrace = {
        x: Array.from({length: 10}, (_, i) => i + 1),
        y: stats.map(s => s.sse),
        mode: 'lines+markers'
    };
    const silhouetteTrace = {
        x: Array.from({length: 10}, (_, i) => i + 1),
        y: stats.map(s => s.silhouette),
        mode: 'lines+markers'
    };

    Plotly.newPlot(elbowPlotDiv, [elbowTrace, {x:[k], y:[stats[k-1].sse], mode:'markers', marker:{color:'red', size:12}}], {title:'Elbow Method', yaxis:{title:'SSE'}, margin:{t:30,r:10,b:20,l:40}});
    Plotly.newPlot(silhouettePlotDiv, [silhouetteTrace, {x:[k], y:[stats[k-1].silhouette], mode:'markers', marker:{color:'red', size:12}}], {title:'Silhouette Score', yaxis:{title:'Score'}, margin:{t:30,r:10,b:20,l:40}});
}

kSlider.addEventListener('input', updateChooseKPlots);
updateChooseKPlots();

});
</script>