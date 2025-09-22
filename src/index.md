---
layout: base.njk
title: "BIT07 Machine Learning Course"
---

<!-- Hero Section -->
<div class="relative overflow-hidden bg-gradient-to-br from-primary-50 via-white to-secondary-50 rounded-3xl mb-12 p-8 md:p-12">
    <div class="relative z-10">
        <div class="text-center mb-8">
            <h1 class="text-4xl md:text-5xl lg:text-6xl font-bold gradient-text mb-6">Machine Learning for Bioinformatics</h1>
            <p class="text-lg md:text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">Master machine learning fundamentals with hands-on bioinformatics applications. From linear models to neural networks, learn to build models that extract insights from biological data.</p>
        </div>
        
        <div class="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8">
            <a href="{{ '/introduction/' | url }}" class="btn-primary text-lg px-8 py-4 inline-flex items-center animate-bounce-subtle">
                🚀 Start Learning
                <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path>
                </svg>
            </a>
            <a href="#course-overview" class="btn-secondary text-lg px-8 py-4">
                📚 Course Overview
            </a>
        </div>
    </div>
    
    <!-- Decorative elements -->
    <div class="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-primary-200 to-transparent rounded-full opacity-20 -translate-y-32 translate-x-32"></div>
    <div class="absolute bottom-0 left-0 w-48 h-48 bg-gradient-to-tr from-secondary-200 to-transparent rounded-full opacity-20 translate-y-24 -translate-x-24"></div>
</div>


<!-- Course Chapters -->
<div id="course-structure" class="mb-16">
    <h2 class="text-3xl font-bold text-center gradient-text mb-12">Course Chapters</h2>
    
    <!-- Learning Path Visualization -->
    <div class="relative mb-12">
        <div class="learning-path"></div>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Foundations -->
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">1</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/introduction/' | url }}" class="block">Introduction to Machine Learning</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Learn the fundamentals of machine learning paradigms and understand how ML differs from traditional programming.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full">Learning Paradigms</span>
                            <span class="inline-block bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full">Project Lifecycle</span>
                            <span class="inline-block bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full">Bioinformatics Applications</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">2</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/linear-regression/' | url }}" class="block">Linear Regression</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Model biological relationships using linear regression. Master gradient descent, preprocessing, and regularization.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">Gradient Descent</span>
                            <span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">Regularization</span>
                            <span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">Bias-Variance</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">3</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/logistic-regression/' | url }}" class="block">Logistic Regression</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Move from prediction to classification. Understand the sigmoid function and evaluation metrics.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full">Sigmoid Function</span>
                            <span class="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full">Odds Ratios</span>
                            <span class="inline-block bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full">Multi-class</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">4</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/hyperparameter-tuning/' | url }}" class="block">Hyperparameter Tuning</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Optimize model performance through systematic hyperparameter tuning and cross-validation.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">Grid Search</span>
                            <span class="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">Cross-Validation</span>
                            <span class="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">Model Selection</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">5</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/imbalanced-data/' | url }}" class="block">Imbalanced Data</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Handle datasets with unequal class representation using sampling techniques and appropriate metrics.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">SMOTE</span>
                            <span class="inline-block bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">Resampling</span>
                            <span class="inline-block bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">Evaluation</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">6</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/naive-bayes/' | url }}" class="block">Naïve Bayes Classifiers</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Master probabilistic classifiers based on Bayes' theorem with practical applications.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded-full">Bayes' Theorem</span>
                            <span class="inline-block bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded-full">Text Classification</span>
                            <span class="inline-block bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded-full">Probabilistic</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">7</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/decision-trees/' | url }}" class="block">Decision Trees</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Build interpretable tree-based models with understanding of splitting criteria and pruning.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded-full">Information Gain</span>
                            <span class="inline-block bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded-full">Pruning</span>
                            <span class="inline-block bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded-full">Interpretability</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">8</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/ensemble-learning/' | url }}" class="block">Ensemble Learning</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Harness the power of combining multiple models through bagging, boosting, and stacking.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-pink-100 text-pink-800 text-xs px-2 py-1 rounded-full">Random Forest</span>
                            <span class="inline-block bg-pink-100 text-pink-800 text-xs px-2 py-1 rounded-full">Boosting</span>
                            <span class="inline-block bg-pink-100 text-pink-800 text-xs px-2 py-1 rounded-full">Stacking</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">9</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/unsupervised-learning/' | url }}" class="block">Unsupervised Learning</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Discover hidden patterns through clustering and dimensionality reduction techniques.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-teal-100 text-teal-800 text-xs px-2 py-1 rounded-full">K-means</span>
                            <span class="inline-block bg-teal-100 text-teal-800 text-xs px-2 py-1 rounded-full">PCA</span>
                            <span class="inline-block bg-teal-100 text-teal-800 text-xs px-2 py-1 rounded-full">t-SNE</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">10</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/ml-workflows/' | url }}" class="block">ML Workflows</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Build robust machine learning pipelines handling missing data and mixed data types.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded-full">Pipelines</span>
                            <span class="inline-block bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded-full">Missing Data</span>
                            <span class="inline-block bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded-full">Deployment</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="chapter-card group">
                <div class="flex items-start space-x-4">
                    <div class="chapter-indicator">11</div>
                    <div class="flex-1">
                        <h3 class="text-xl font-bold text-gray-800 group-hover:text-primary-600 transition-colors mb-2">
                            <a href="{{ '/neural-networks/' | url }}" class="block">Neural Networks & Deep Learning</a>
                        </h3>
                        <p class="text-gray-600 mb-3">Introduction to deep learning with feedforward networks, backpropagation, and modern architectures.</p>
                        <div class="flex flex-wrap gap-2">
                            <span class="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full">TensorFlow</span>
                            <span class="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full">Backpropagation</span>
                            <span class="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full">CNNs & RNNs</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Learning Path Guide -->
<div class="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-2xl p-8 mb-16">
    <h2 class="text-2xl font-bold text-center mb-8 gradient-text">🎯 Your Learning Journey</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="text-center">
            <div class="w-16 h-16 bg-primary-500 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">1</div>
            <h3 class="font-semibold text-gray-800 mb-2">Foundations</h3>
            <p class="text-sm text-gray-600">Chapters 1-3: Core concepts, linear models, and classification basics</p>
        </div>
        <div class="text-center">
            <div class="w-16 h-16 bg-secondary-500 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">2</div>
            <h3 class="font-semibold text-gray-800 mb-2">Optimization</h3>
            <p class="text-sm text-gray-600">Chapters 4-5: Model tuning and handling real-world data challenges</p>
        </div>
        <div class="text-center">
            <div class="w-16 h-16 bg-accent-500 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">3</div>
            <h3 class="font-semibold text-gray-800 mb-2">Advanced Models</h3>
            <p class="text-sm text-gray-600">Chapters 6-8: Probabilistic models, trees, and ensemble methods</p>
        </div>
        <div class="text-center">
            <div class="w-16 h-16 bg-primary-700 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">4</div>
            <h3 class="font-semibold text-gray-800 mb-2">Specialized Topics</h3>
            <p class="text-sm text-gray-600">Chapters 9-11: Unsupervised learning, workflows, and neural networks</p>
        </div>
    </div>
    
    <div class="mt-8 text-center">
        <div class="inline-flex items-center space-x-2 bg-white bg-opacity-50 rounded-full px-6 py-3">
            <svg class="w-5 h-5 text-primary-600" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path>
            </svg>
            <span class="text-sm font-medium text-gray-700">💡 <strong>Pro Tip:</strong> Start with Chapter 1 and progress sequentially. Each chapter builds upon previous concepts with practical Python implementations.</span>
        </div>
    </div>
</div>
