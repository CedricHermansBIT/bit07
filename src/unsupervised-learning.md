---
layout: base.njk
title: "Unsupervised Learning"
---

<div class="content-section">
    <h2>What is Unsupervised Learning?</h2>
    <p>Unsupervised learning discovers hidden patterns in data without labeled examples. Unlike supervised learning, we don't have target values to predict. Instead, we explore the data's inherent structure to find groupings, reduce dimensions, or detect anomalies.</p>
    
    <div class="alert alert-info">
        <strong>Key Idea:</strong> Learn patterns from data structure alone, without knowing the "correct" answers.
    </div>
</div>

<div class="content-section">
    <h2>Types of Unsupervised Learning</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-blue-50 rounded-lg p-4">
            <h3 class="text-lg font-semibold text-blue-800 mb-2">Clustering</h3>
            <p class="text-blue-700">Group similar data points together to discover natural categories in the data.</p>
        </div>
        <div class="bg-green-50 rounded-lg p-4">
            <h3 class="text-lg font-semibold text-green-800 mb-2">Dimensionality Reduction</h3>
            <p class="text-green-700">Reduce the number of features while preserving important information for visualization or preprocessing.</p>
        </div>
    </div>
</div>

<div class="content-section">
    <h2>Clustering: Finding Groups in Data</h2>
    <p>Clustering algorithms group similar data points together without prior knowledge of group labels. This is essential for discovering biological subtypes, patient populations, or gene co-expression modules.</p>

    <h3>K-Means Clustering</h3>
    <h4>Algorithm Steps</h4>
    <ol>
        <li>Choose the number of clusters (k)</li>
        <li>Initialize k centroids randomly</li>
        <li>Assign each point to the nearest centroid</li>
        <li>Update centroids to the mean of assigned points</li>
        <li>Repeat steps 3-4 until convergence</li>
    </ol>
    
    <h3>Hierarchical Clustering</h3>
    <h4>Agglomerative (Bottom-Up)</h4>
    <ol>
        <li>Start with each point as its own cluster</li>
        <li>Merge the closest pair of clusters</li>
        <li>Repeat until all points are in one cluster</li>
        <li>Cut the dendrogram at desired level</li>
    </ol>
    
    <h3>DBSCAN (Density-Based)</h3>
    <p>Groups points that are closely packed while marking outliers as noise. Excellent for finding clusters of arbitrary shapes.</p>
</div>

<!-- Interactive Clustering Demo -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4">🔬 Interactive Demo: Gene Expression Clustering</h3>
    <p class="text-gray-600 mb-4">Explore how different clustering algorithms group gene expression data:</p>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Clustering Algorithm</label>
                    <select id="cluster-algorithm" class="w-full p-2 border border-gray-300 rounded-lg">
                        <option value="kmeans">K-Means</option>
                        <option value="hierarchical">Hierarchical</option>
                        <option value="dbscan">DBSCAN</option>
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Number of Clusters</label>
                    <input type="range" id="num-clusters" min="2" max="8" value="3" class="parameter-slider">
                    <span id="clusters-value" class="text-sm text-gray-600">3</span>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Data Noise Level</label>
                    <input type="range" id="noise-level" min="0" max="1" step="0.1" value="0.2" class="parameter-slider">
                    <span id="noise-level-value" class="text-sm text-gray-600">0.2</span>
                </div>
                
                <button id="generate-cluster-data" class="btn-primary w-full">🔄 Generate New Data</button>
            </div>
        </div>
        
        <div>
            <div id="clustering-plot" style="width:100%;height:300px;"></div>
        </div>
    </div>
    
    <div class="mt-4 p-3 bg-gray-50 rounded-lg">
        <p class="text-sm text-gray-600" id="cluster-info">Clustering 100 genes into 3 groups based on expression patterns across different conditions.</p>
    </div>
</div>

<div class="content-section">
    <h2>Dimensionality Reduction</h2>
    <p>Dimensionality reduction transforms high-dimensional data to lower dimensions while preserving important patterns. This is crucial for visualization and preprocessing of biological data.</p>

    <h3>Principal Component Analysis (PCA)</h3>
    <h4>How PCA Works</h4>
    <ol>
        <li>Standardize the data (mean = 0, variance = 1)</li>
        <li>Calculate covariance matrix</li>
        <li>Compute eigenvalues and eigenvectors</li>
        <li>Sort by eigenvalues (descending)</li>
        <li>Select top k eigenvectors (principal components)</li>
        <li>Transform data to new coordinate system</li>
    </ol>
    
    <h3>t-SNE (t-Distributed Stochastic Neighbor Embedding)</h3>
    <p>Excellent for visualizing high-dimensional data in 2D or 3D by preserving local neighborhoods.</p>
    
    <h3>UMAP (Uniform Manifold Approximation and Projection)</h3>
    <p>Modern alternative to t-SNE with better global structure preservation and faster computation.</p>
</div>

<!-- Interactive PCA Demo -->
<div class="bg-white rounded-2xl shadow-lg p-6 mb-8 border border-gray-100">
    <h3 class="text-xl font-bold text-gray-800 mb-4">📊 Interactive Demo: PCA Visualization</h3>
    <p class="text-gray-600 mb-4">See how PCA reduces dimensionality while preserving variance:</p>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Original Dimensions</label>
                    <input type="range" id="orig-dims" min="3" max="10" value="5" class="parameter-slider">
                    <span id="orig-dims-value" class="text-sm text-gray-600">5</span>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Reduced Dimensions</label>
                    <input type="range" id="reduced-dims" min="2" max="3" value="2" class="parameter-slider">
                    <span id="reduced-dims-value" class="text-sm text-gray-600">2</span>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Dataset Type</label>
                    <select id="pca-dataset" class="w-full p-2 border border-gray-300 rounded-lg">
                        <option value="gene-expression">Gene Expression</option>
                        <option value="protein-levels">Protein Levels</option>
                        <option value="metabolomics">Metabolomics</option>
                    </select>
                </div>
                
                <button id="run-pca" class="btn-primary w-full">🎯 Run PCA</button>
            </div>
            
            <div class="mt-4 p-3 bg-blue-50 rounded-lg">
                <h4 class="font-semibold text-blue-800 mb-2">Explained Variance</h4>
                <div id="variance-explained" class="space-y-1">
                    <div class="flex justify-between text-sm">
                        <span>PC1:</span>
                        <span id="pc1-var">45.2%</span>
                    </div>
                    <div class="flex justify-between text-sm">
                        <span>PC2:</span>
                        <span id="pc2-var">28.7%</span>
                    </div>
                    <div class="flex justify-between text-sm font-semibold">
                        <span>Total:</span>
                        <span id="total-var">73.9%</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div>
            <div id="pca-plot" style="width:100%;height:300px;"></div>
        </div>
    </div>
</div>

<div class="content-section">
    <h2>Evaluation Metrics</h2>
    <h3>For Clustering (Internal Metrics)</h3>
    <ul>
        <li><strong>Silhouette Score:</strong> Measures how similar points are within clusters vs. other clusters</li>
        <li><strong>Calinski-Harabasz Index:</strong> Ratio of between-cluster to within-cluster variance</li>
        <li><strong>Davies-Bouldin Index:</strong> Average similarity between clusters</li>
    </ul>
    
    <h3>For Dimensionality Reduction</h3>
    <ul>
        <li><strong>Explained Variance Ratio:</strong> How much variance each component captures</li>
        <li><strong>Reconstruction Error:</strong> Difference between original and reconstructed data</li>
        <li><strong>Trustworthiness:</strong> Whether nearby points in low-D were nearby in high-D</li>
    </ul>
</div>

<div class="content-section">
    <h2>Biological Applications</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
            <h4 class="font-semibold text-blue-800 mb-2">🧬 Single-Cell Analysis</h4>
            <p class="text-sm text-blue-700">Cluster cells by type and visualize with t-SNE/UMAP</p>
        </div>
        
        <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4">
            <h4 class="font-semibold text-green-800 mb-2">🔬 Gene Co-expression</h4>
            <p class="text-sm text-green-700">Find modules of co-regulated genes</p>
        </div>
        
        <div class="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4">
            <h4 class="font-semibold text-purple-800 mb-2">🧪 Drug Discovery</h4>
            <p class="text-sm text-purple-700">Cluster compounds by similarity</p>
        </div>
        
        <div class="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4">
            <h4 class="font-semibold text-red-800 mb-2">🏥 Patient Stratification</h4>
            <p class="text-sm text-red-700">Identify disease subtypes</p>
        </div>
    </div>
</div>

<div class="highlight">
    <strong>When to Use:</strong> Use clustering for discovering natural groupings in data (cell types, patient subtypes) and dimensionality reduction for visualization, noise reduction, or preprocessing high-dimensional biological data before applying supervised learning.
</div>

