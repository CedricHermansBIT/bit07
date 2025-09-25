---
layout: base.njk
title: "Chapter 11: Neural Networks and Deep Learning"
---

<!-- Header -->
<div class="bg-gradient-to-r from-gray-50 to-slate-100 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 11: Neural Networks & Deep Learning</h2>
    <p class="text-gray-700 leading-relaxed">While models like Random Forests are powerful, they can be limited in their ability to capture the extraordinarily complex patterns in biological data (e.g., predicting protein structure from a sequence). To tackle these challenges, we turn to <strong>Artificial Neural Networks (ANNs)</strong>, the core of deep learning.</p>
</div>

<!-- 1. The Perceptron -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">The Building Block: A Single Artificial Neuron</h3>
    <p class="text-gray-700 mb-4">The simplest neural network is a <strong>Perceptron</strong>, which is just a single neuron. It takes multiple inputs, computes a weighted sum, and if that sum exceeds a threshold, it "fires." This simple unit is a linear classifier, much like logistic regression.</p>

    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <p class="text-sm text-gray-600 mb-4">Adjust the weights and bias to move the decision boundary and correctly classify the two gene types.</p>
            <div class="space-y-3">
                <div>
                    <label class="block text-sm font-medium">Weight 1 (for Feature X)</label>
                    <input id="p-w1" type="range" min="-5" max="5" step="0.1" value="1" class="parameter-slider">
                </div>
                 <div>
                    <label class="block text-sm font-medium">Weight 2 (for Feature Y)</label>
                    <input id="p-w2" type="range" min="-5" max="5" step="0.1" value="1" class="parameter-slider">
                </div>
                 <div>
                    <label class="block text-sm font-medium">Bias</label>
                    <input id="p-bias" type="range" min="-5" max="5" step="0.1" value="0" class="parameter-slider">
                </div>
            </div>
        </div>
        <div>
            <div id="perceptron-plot" style="width:100%;height:300px;"></div>
        </div>
    </div>
</div>

<!-- 2. Feedforward Networks -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">From One Neuron to Many: Feedforward Networks</h3>
    <p class="text-gray-700 mb-4">A single neuron can only learn linear boundaries. The real power comes from organizing neurons into layers: an <strong>Input Layer</strong>, one or more <strong>Hidden Layers</strong>, and an <strong>Output Layer</strong>. Information flows forward through the network. A network with two or more hidden layers is considered a <strong>"deep" neural network</strong>.</p>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: The Power of Hidden Layers</h4>
    <p class="text-gray-600 mb-4">This dataset cannot be separated by a single line. Add hidden layers and neurons to see how the network learns a more complex, non-linear decision boundary.</p>

    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <div class="space-y-4">
                 <div>
                    <label class="block text-sm font-medium">Hidden Layers</label>
                    <div class="flex gap-2 items-center">
                         <button id="rem-layer" class="btn-secondary px-3 py-1 text-lg">-</button>
                         <span id="layers-display" class="font-mono text-center w-24">1 Layer</span>
                         <button id="add-layer" class="btn-secondary px-3 py-1 text-lg">+</button>
                    </div>
                </div>
                 <div>
                    <label class="block text-sm font-medium">Neurons per Layer</label>
                    <input id="neurons-slider" type="range" min="2" max="10" step="1" value="3" class="parameter-slider">
                    <span id="neurons-display" class="text-sm">3 Neurons</span>
                </div>
                <button id="train-network-btn" class="btn-primary w-full">Train Network</button>
            </div>
        </div>
        <div>
             <div id="network-plot" style="width:100%;height:300px;"></div>
        </div>
    </div>
</div>

<!-- 3. Backpropagation -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">How Networks Learn: Backpropagation</h3>
    <p class="text-gray-700 mb-4">The network learns the correct weights through an algorithm called <strong>backpropagation</strong>, which is essentially a clever application of Gradient Descent. It's a cycle that repeats for many "epochs" (passes over the entire dataset).</p>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-center text-sm">
        <div class="bg-blue-50 p-3 rounded-lg border border-blue-200">
            <div class="text-2xl">1️⃣</div> <strong class="text-blue-800">Forward Pass</strong>
            <p>Input data is fed forward through the network to produce a prediction.</p>
        </div>
         <div class="bg-red-50 p-3 rounded-lg border border-red-200">
            <div class="text-2xl">2️⃣</div> <strong class="text-red-800">Calculate Loss</strong>
            <p>The prediction is compared to the true label to quantify the error (loss).</p>
        </div>
         <div class="bg-purple-50 p-3 rounded-lg border border-purple-200">
            <div class="text-2xl">3️⃣</div> <strong class="text-purple-800">Backward Pass</strong>
            <p>The loss is propagated backward to calculate how much each weight contributed to the error.</p>
        </div>
        <div class="bg-green-50 p-3 rounded-lg border border-green-200">
            <div class="text-2xl">4️⃣</div> <strong class="text-green-800">Update Weights</strong>
            <p>Weights are adjusted slightly in the direction that reduces the loss.</p>
        </div>
    </div>
</div>

<!-- 4. Activation Functions -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">Activation Functions: The Neuron's "Firing" Mechanism</h3>
    <p class="text-gray-700 mb-4">Activation functions introduce essential <strong>non-linearity</strong> into the network. Without them, even a deep network would behave like a simple linear model.</p>
    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <label class="block text-sm font-medium">Select Function:</label>
            <select id="activation-select" class="w-full p-2 border rounded mt-1">
                <option value="relu" selected>ReLU (Rectified Linear Unit)</option>
                <option value="leaky_relu">Leaky ReLU</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
            </select>
            <div id="activation-info" class="text-sm mt-3 p-3 bg-white rounded-lg"></div>
        </div>
        <div>
            <div id="activation-plot" style="width:100%;height:250px;"></div>
        </div>
    </div>
</div>

<!-- 5. Key Components -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-2">Key Components Explained</h3>
    <p class="text-gray-700 mb-6">When training a neural network, these API pieces work together. Think of it as: <em>assemble layers ➜ choose how they transform data ➜ configure learning ➜ run training</em>.</p>

    <!-- High-Level Flow Diagram -->
    <div class="relative overflow-x-auto mb-8" aria-label="Neural network training flow">
        <div class="flex items-stretch gap-3 min-w-[780px] text-sm font-medium">
            <div class="flex-1 bg-gradient-to-br from-sky-50 to-sky-100 border border-sky-200 rounded-lg p-4 flex flex-col items-center justify-center">
                <span class="text-sky-700 font-semibold mb-1">1. Data</span>
                <span class="text-sky-600 text-xs text-center">Prepared & batched input features</span>
            </div>
            <div class="flex items-center text-gray-400 px-1">➡</div>
            <div class="flex-1 bg-gradient-to-br from-indigo-50 to-indigo-100 border border-indigo-200 rounded-lg p-4 flex flex-col items-center justify-center">
                <span class="text-indigo-700 font-semibold mb-1">2. Layers</span>
                <span class="text-indigo-600 text-xs text-center">Dense + activations (forward pass)</span>
            </div>
            <div class="flex items-center text-gray-400 px-1">➡</div>
            <div class="flex-1 bg-gradient-to-br from-rose-50 to-rose-100 border border-rose-200 rounded-lg p-4 flex flex-col items-center justify-center">
                <span class="text-rose-700 font-semibold mb-1">3. Loss</span>
                <span class="text-rose-600 text-xs text-center">Compare predictions vs labels</span>
            </div>
            <div class="flex items-center text-gray-400 px-1">➡</div>
            <div class="flex-1 bg-gradient-to-br from-amber-50 to-amber-100 border border-amber-200 rounded-lg p-4 flex flex-col items-center justify-center">
                <span class="text-amber-700 font-semibold mb-1">4. Optimizer</span>
                <span class="text-amber-600 text-xs text-center">Uses gradients to update weights</span>
            </div>
            <div class="flex items-center text-gray-400 px-1">➡</div>
            <div class="flex-1 bg-gradient-to-br from-emerald-50 to-emerald-100 border border-emerald-200 rounded-lg p-4 flex flex-col items-center justify-center">
                <span class="text-emerald-700 font-semibold mb-1">5. Improved Model</span>
                <span class="text-emerald-600 text-xs text-center">Repeat for many epochs</span>
            </div>
        </div>
        <div class="absolute -bottom-3 left-1/2 -translate-x-1/2 bg-white px-3 py-1 rounded-full shadow border text-[11px] text-gray-600">Forward Pass ➜ Loss ➜ Backpropagation ➜ Weight Update</div>
    </div>

    <dl class="divide-y divide-gray-200">
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">1</span>Sequential API</dt>
            <dd class="text-sm text-gray-700">Simplest way to build a model: add layers in order (a straight stack). Great for most feedforward networks.</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">2</span>Dense Layer</dt>
            <dd class="text-sm text-gray-700">Fully connected layer; every neuron sees all outputs from the previous layer. <code>units</code> = number of neurons.</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">3</span>Activation Functions</dt>
            <dd class="text-sm text-gray-700">Inject non‑linearity so the network can model complex patterns. Hidden layers: <strong>ReLU</strong>; output (binary): <strong>Sigmoid</strong>.</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">4</span>Compile Method</dt>
            <dd class="text-sm text-gray-700">Locks in the <em>optimizer</em>, <em>loss</em>, and desired <em>metrics</em> so training knows what to minimize & report.</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">5</span>Optimizer</dt>
            <dd class="text-sm text-gray-700"><code>Adam</code> adaptively tunes learning using estimates of first & second moments of gradients. Strong default.</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">6</span>Loss Function</dt>
            <dd class="text-sm text-gray-700"><code>binaryCrossentropy</code> compares predicted probabilities vs. true labels (binary). Others: <code>mse</code> (regression), <code>categoricalCrossentropy</code> (multi‑class).</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">7</span>Metrics</dt>
            <dd class="text-sm text-gray-700">Quick feedback (e.g. <strong>accuracy</strong>). For imbalance add precision, recall, F1 (can compute from confusion matrix).</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">8</span>Fit Method</dt>
            <dd class="text-sm text-gray-700">Runs training for N <em>epochs</em>; optional <code>validationSplit</code> or callbacks (early stopping, logging, etc.).</dd>
        </div>
        <div class="grid md:grid-cols-[190px_1fr] gap-4 py-3">
            <dt class="font-semibold text-gray-800 flex items-center gap-2"><span class="inline-flex items-center justify-center w-6 h-6 rounded bg-indigo-100 text-indigo-600 text-xs font-bold">9</span>Epochs & Batch Size</dt>
            <dd class="text-sm text-gray-700"><strong>Epoch</strong>: one pass over all data. <strong>Batch size</strong>: samples per gradient update; smaller = noisier but can generalize better.</dd>
        </div>
    </dl>

    <div class="mt-6 p-4 rounded-lg bg-gray-50 border border-gray-200">
        <h4 class="font-semibold text-gray-800 mb-2 text-sm tracking-wide uppercase">Sequential vs Functional (Advanced)</h4>
        <pre class="text-xs leading-relaxed overflow-x-auto"><code>// Sequential API (single-input, single-path models)
const modelSeq = tf.sequential();
modelSeq.add(tf.layers.dense({units:16, activation:'relu', inputShape:[10]}));
modelSeq.add(tf.layers.dense({units:1, activation:'sigmoid'}));
modelSeq.compile({optimizer:'adam', loss:'binaryCrossentropy'});

// Functional API (flexible graphs / multi-input / shared layers)
const inputs = tf.input({shape:[10]});
const hidden = tf.layers.dense({units:16, activation:'relu'}).apply(inputs);
const outputs = tf.layers.dense({units:1, activation:'sigmoid'}).apply(hidden);
const modelFunc = tf.model({inputs, outputs});</code></pre>
        <p class="mt-3 text-xs text-gray-600">Use the Functional API when you need branching, skip connections, shared layers, or multiple inputs/outputs.</p>
    </div>
</div>

<!-- 6. Advanced Architectures -->
<div class="card mb-8" id="advanced-architectures">
    <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-4">
        <h3 class="text-2xl font-bold text-gray-800 dark:text-gray-100">Beyond Feedforward: Advanced Architectures</h3>
        <span class="inline-flex items-center gap-2 text-xs font-semibold tracking-wide bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 px-3 py-1 rounded-full">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="opacity-80"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            ARCHITECTURE LANDSCAPE
        </span>
    </div>
    <p class="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed">Different architectures exploit different structural biases: locality (CNN), temporal order (RNN), global relational context (Transformers), low‑dimensional manifolds (Autoencoders), and explicit relational graphs (GNNs). These inductive biases often matter more than raw depth.</p>

    <!-- Architecture Cards Grid -->
    <div class="grid md:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
        <!-- CNN Card -->
        <div class="group bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-sm hover:shadow-md transition" aria-labelledby="cnn-title">
            <div class="flex items-start justify-between mb-3">
                <h4 id="cnn-title" class="font-semibold text-gray-800 dark:text-gray-100">Convolutional Neural Networks (CNNs)</h4>
                <span class="text-[10px] font-bold bg-sky-100 dark:bg-sky-900/40 text-sky-700 dark:text-sky-300 px-2 py-1 rounded">LOCAL</span>
            </div>
            <div class="h-32 flex items-center justify-center">
                <!-- Improved CNN schematic -->
                <img src="https://www.researchgate.net/publication/336805909/figure/fig1/AS:11431281342674890@1743562251174/Schematic-diagram-of-a-basic-convolutional-neural-network-CNN-architecture-26.tif" class="w-full max-w-[260px]" aria-label="CNN visualisation">
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">Learns translation‑robust local motifs using shared filters; depth builds hierarchical features.</p>
            <ul class="text-xs text-gray-700 dark:text-gray-300 space-y-1 mb-2 list-disc ml-4">
                <li>Motif discovery (promoters, enhancers)</li>
                <li>Microscopy phenotyping</li>
                <li>Signal denoising</li>
            </ul>
            <div class="text-[11px] text-sky-700 dark:text-sky-300 font-medium">Bias: locality + weight sharing.</div>
        </div>
        <!-- RNN Card -->
        <div class="group bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-sm hover:shadow-md transition" aria-labelledby="rnn-title">
            <div class="flex items-start justify-between mb-3">
                <h4 id="rnn-title" class="font-semibold text-gray-800 dark:text-gray-100">Recurrent Neural Networks (RNNs)</h4>
                <span class="text-[10px] font-bold bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 px-2 py-1 rounded">SEQUENCE</span>
            </div>
            <div class="h-32 flex items-center justify-center">
                <!-- Improved RNN schematic -->
                <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250523171309383561/recurrent_neural_network.webp" class="w-full max-w-[260px]" aria-label="RNN visualisation">
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">Hidden state carries temporal context; gated variants (LSTM/GRU) mitigate vanishing gradients.</p>
            <ul class="text-xs text-gray-700 dark:text-gray-300 space-y-1 mb-2 list-disc ml-4">
                <li>Protein secondary structure</li>
                <li>Time‑course expression profiles</li>
                <li>Signal segmentation</li>
            </ul>
            <div class="text-[11px] text-amber-700 dark:text-amber-300 font-medium">Bias: ordered temporal dependence.</div>
        </div>
        <!-- Transformer Card -->
        <div class="group bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-sm hover:shadow-md transition" aria-labelledby="trf-title">
            <div class="flex items-start justify-between mb-3">
                <h4 id="trf-title" class="font-semibold text-gray-800 dark:text-gray-100">Transformers</h4>
                <span class="text-[10px] font-bold bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 px-2 py-1 rounded">ATTENTION</span>
            </div>
            <div class="h-32 flex items-center justify-center">
                <!-- Improved Transformer schematic -->
                <img src="https://deeprevision.github.io/posts/001-transformer/transformer.png" class="w-full max-w-[260px]" aria-label="Transformer visualisation">
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">Self‑attention builds pairwise relationships; residual + layer norm stabilize deep stacking.</p>
            <ul class="text-xs text-gray-700 dark:text-gray-300 space-y-1 mb-2 list-disc ml-4">
                <li>Protein folding (AlphaFold2)</li>
                <li>Long sequence embeddings</li>
                <li>Variant effect modeling</li>
            </ul>
            <div class="text-[11px] text-purple-700 dark:text-purple-300 font-medium">Bias: global token interactions.</div>
        </div>
        <!-- Autoencoder Card -->
        <div class="group bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-sm hover:shadow-md transition" aria-labelledby="ae-title">
            <div class="flex items-start justify-between mb-3">
                <h4 id="ae-title" class="font-semibold text-gray-800 dark:text-gray-100">Autoencoders</h4>
                <span class="text-[10px] font-bold bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300 px-2 py-1 rounded">LATENT</span>
            </div>
            <div class="h-32 flex items-center justify-center">
                <img src="https://www.researchgate.net/publication/317559243/figure/fig2/AS:531269123805186@1503675837486/Deep-Autoencoder-DAE.png" class="max-w-[260px] max-h-32" aria-label="Autoencoder visualisation">
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">Learns compressed latent representation; variants (VAE, denoising) add probabilistic or robustness constraints.</p>
            <ul class="text-xs text-gray-700 dark:text-gray-300 space-y-1 mb-2 list-disc ml-4">
                <li>Single‑cell embedding</li>
                <li>Denoising expression matrices</li>
                <li>Anomaly detection</li>
            </ul>
            <div class="text-[11px] text-teal-700 dark:text-teal-300 font-medium">Bias: information bottleneck.</div>
        </div>
        <!-- GNN Card -->
        <div class="group bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl p-5 shadow-sm hover:shadow-md transition" aria-labelledby="gnn-title">
            <div class="flex items-start justify-between mb-3">
                <h4 id="gnn-title" class="font-semibold text-gray-800 dark:text-gray-100">Graph Neural Networks (GNNs)</h4>
                <span class="text-[10px] font-bold bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-300 px-2 py-1 rounded">RELATIONAL</span>
            </div>
            <div class="h-32 flex items-center justify-center">
                <img src="https://www.techtarget.com/rms/onlineimages/the_structure_of_a_graph_neural_network-f_mobile.png" class="max-w-[260px] max-h-32" aria-label="GNN visualisation">
            </div>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-3">Message passing updates node embeddings using neighbor information; stacking expands receptive field.</p>
            <ul class="text-xs text-gray-700 dark:text-gray-300 space-y-1 mb-2 list-disc ml-4">
                <li>Protein interaction networks</li>
                <li>Pathway / metabolic graphs</li>
                <li>Drug–target link prediction</li>
            </ul>
            <div class="text-[11px] text-rose-700 dark:text-rose-300 font-medium">Bias: relational locality.</div>
        </div>
    </div>

    <!-- Comparison Table -->
    <div class="overflow-x-auto mb-8">
        <table class="w-full text-sm text-left border border-gray-200 dark:border-slate-700 rounded-lg overflow-hidden">
            <thead class="bg-gray-50 dark:bg-slate-800/70 text-gray-700 dark:text-gray-300 font-semibold">
                <tr>
                    <th class="px-4 py-2">Architecture</th>
                    <th class="px-4 py-2">Core Strength</th>
                    <th class="px-4 py-2">Bioinformatics Use Case</th>
                    <th class="px-4 py-2">Primary Limitation</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-slate-700">
                <tr class="hover:bg-gray-50 dark:hover:bg-slate-800/60">
                    <td class="px-4 py-2 font-medium">CNN</td>
                    <td class="px-4 py-2">Local motif extraction</td>
                    <td class="px-4 py-2">Motif scanning; phenotyping</td>
                    <td class="px-4 py-2 text-xs">Limited long‑range context</td>
                </tr>
                <tr class="hover:bg-gray-50 dark:hover:bg-slate-800/60">
                    <td class="px-4 py-2 font-medium">RNN (LSTM/GRU)</td>
                    <td class="px-4 py-2">Sequential memory</td>
                    <td class="px-4 py-2">Temporal expression; sequence labeling</td>
                    <td class="px-4 py-2 text-xs">Hard to parallelize; long sequences degrade</td>
                </tr>
                <tr class="hover:bg-gray-50 dark:hover:bg-slate-800/60">
                    <td class="px-4 py-2 font-medium">Transformer</td>
                    <td class="px-4 py-2">Global attention</td>
                    <td class="px-4 py-2">Protein folding; embeddings</td>
                    <td class="px-4 py-2 text-xs">Quadratic memory (length²)</td>
                </tr>
                <tr class="hover:bg-gray-50 dark:hover:bg-slate-800/60">
                    <td class="px-4 py-2 font-medium">Autoencoder</td>
                    <td class="px-4 py-2">Latent compression</td>
                    <td class="px-4 py-2">Single‑cell reduction; denoising</td>
                    <td class="px-4 py-2 text-xs">Latent interpretability</td>
                </tr>
                <tr class="hover:bg-gray-50 dark:hover:bg-slate-800/60">
                    <td class="px-4 py-2 font-medium">GNN</td>
                    <td class="px-4 py-2">Relational reasoning</td>
                    <td class="px-4 py-2">Interaction networks; pathways</td>
                    <td class="px-4 py-2 text-xs">Oversmoothing / scaling</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Guidance -->
    <details class="group mb-4 border border-gray-200 dark:border-slate-700 rounded-lg p-4 bg-white dark:bg-slate-800">
        <summary class="cursor-pointer font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
            <span class="w-5 h-5 inline-flex items-center justify-center rounded bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-300 text-xs font-bold group-open:rotate-90 transition-transform">+</span>
            Choosing the right architecture
        </summary>
        <div class="mt-3 text-sm text-gray-700 dark:text-gray-300 space-y-2">
            <p><strong>CNN</strong>: Structured local patterns (motifs, localized morphology).</p>
            <p><strong>RNN</strong>: Moderate length ordered signals where stepwise dependencies matter.</p>
            <p><strong>Transformer</strong>: Long sequences or tasks needing global pairwise context; scale with data.</p>
            <p><strong>Autoencoder</strong>: Representation learning, noise reduction, anomaly detection.</p>
            <p><strong>GNN</strong>: Interacting entities (proteins, metabolites, drugs) with graph edges encoding biology.</p>
        </div>
    </details>

    <p class="text-xs text-gray-500 dark:text-gray-400">Modern systems are often hybrids (e.g., CNN front‑ends feeding Transformers, graph‑aware attention blocks, or variational autoencoder cores in generative pipelines).</p>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- 1. PERCEPTRON DEMO ---
    const p_w1 = document.getElementById('p-w1');
    const p_w2 = document.getElementById('p-w2');
    const p_bias = document.getElementById('p-bias');
    const perceptronPlot = document.getElementById('perceptron-plot');
    const p_data = {
        x: Array.from({length: 100}, () => Math.random() * 10 - 5),
        y: Array.from({length: 100}, () => Math.random() * 10 - 5),
        labels: []
    };
    p_data.labels = p_data.x.map((x, i) => (x + p_data.y[i] > 0 ? 1 : 0));

    function updatePerceptronPlot() {
        const w1 = parseFloat(p_w1.value);
        const w2 = parseFloat(p_w2.value);
        const bias = parseFloat(p_bias.value);

        const xLine = [-5, 5];
        const yLine = xLine.map(x => (-bias - w1 * x) / w2);

        Plotly.newPlot(perceptronPlot, [{
            x: p_data.x, y: p_data.y, mode: 'markers', type: 'scatter',
            marker: { color: p_data.labels.map(l => l === 1 ? '#60a5fa' : '#f97316') }
        }, {
            x: xLine, y: yLine, mode: 'lines', line: { color: 'black', width: 3 }
        }], { title: 'Perceptron Decision Boundary', showlegend: false, xaxis:{range:[-5,5]}, yaxis:{range:[-5,5]}, margin:{t:40,r:10,b:20,l:20} });
    }
    [p_w1, p_w2, p_bias].forEach(el => el.addEventListener('input', updatePerceptronPlot));
    updatePerceptronPlot();
    
// --- 2. NETWORK BUILDER DEMO ---
const trainBtn = document.getElementById('train-network-btn');
const networkPlot = document.getElementById('network-plot');
const addLayerBtn = document.getElementById('add-layer');
const remLayerBtn = document.getElementById('rem-layer');
const layersDisplay = document.getElementById('layers-display');
const neuronsSlider = document.getElementById('neurons-slider');
const neuronsDisplay = document.getElementById('neurons-display');
let hiddenLayers = 1;
let isTraining = false;

function updateNetworkBuilderUI() {
    layersDisplay.textContent = hiddenLayers + ' Layer' + (hiddenLayers > 1 ? 's' : '');
    neuronsDisplay.textContent = neuronsSlider.value + ' Neurons';
    remLayerBtn.disabled = hiddenLayers === 1;
    addLayerBtn.disabled = hiddenLayers === 4;
}

addLayerBtn.addEventListener('click', () => { hiddenLayers++; updateNetworkBuilderUI(); });
remLayerBtn.addEventListener('click', () => { hiddenLayers--; updateNetworkBuilderUI(); });
neuronsSlider.addEventListener('input', updateNetworkBuilderUI);

// Generate circular dataset (inner circle = class 0, outer ring = class 1)
const nn_data = { x: [], y: [], labels: [] };
for (let i = 0; i < 200; i++) {
    const r = Math.random() * 4;
    const theta = Math.random() * 2 * Math.PI;
    nn_data.x.push(r * Math.sin(theta));
    nn_data.y.push(r * Math.cos(theta));
    nn_data.labels.push(r > 2 ? 1 : 0);
}

// Plot initial data
function plotNetworkBoundary(predictionFunc = null) {
    const traces = [{
        x: nn_data.x, 
        y: nn_data.y, 
        mode: 'markers', 
        type: 'scatter',
        marker: { 
            color: nn_data.labels.map(l => l === 1 ? '#60a5fa' : '#f97316'),
            size: 8
        }
    }];
    
    // If we have a prediction function, add the decision boundary
    if (predictionFunc) {
        // Create grid for decision boundary
        const resolution = 50;
        const x_range = Array.from({length: resolution+1}, (_, i) => (i/resolution)*10-5);
        const y_range = Array.from({length: resolution+1}, (_, i) => (i/resolution)*10-5);
        
        // Generate predictions for each point in the grid
        let z = [];
        for (let i = 0; i < y_range.length; i++) {
            let row = [];
            for (let j = 0; j < x_range.length; j++) {
                row.push(predictionFunc(x_range[j], y_range[i]));
            }
            z.push(row);
        }
        
        traces.push({
            z: z,
            x: x_range,
            y: y_range,
            type: 'contour',
            showscale: false,
            colorscale: [['0', 'rgba(249,115,22,0.8)'], ['1', 'rgba(96,165,250,0.8)']],
            contours: { start: 0.5, end: 0.5, coloring: 'lines' }
        });
    }
    
    Plotly.newPlot(networkPlot, traces, { 
        title: 'Network Decision Boundary', 
        showlegend: false, 
        xaxis: {range: [-5, 5]}, 
        yaxis: {range: [-5, 5]}, 
        margin: {t: 40, r: 10, b: 20, l: 20} 
    });
}

// Initial plot without decision boundary
plotNetworkBoundary();

// Precompute grid and fast decision boundary rendering helpers
const GRID_RES = 40;
const x_grid = Array.from({ length: GRID_RES + 1 }, (_, i) => (i / GRID_RES) * 10 - 5);
const y_grid = Array.from({ length: GRID_RES + 1 }, (_, i) => (i / GRID_RES) * 10 - 5);

function computeDecisionZForModel(model) {
    return tf.tidy(() => {
        const points = [];
        for (let yi = 0; yi < y_grid.length; yi++) {
            for (let xi = 0; xi < x_grid.length; xi++) {
                points.push([x_grid[xi], y_grid[yi]]);
            }
        }
        const gridTensor = tf.tensor2d(points);
        const preds = model.predict(gridTensor);
        const flat = preds.dataSync(); // 1D typed array
        const z = [];
        for (let yi = 0; yi < y_grid.length; yi++) {
            const row = [];
            for (let xi = 0; xi < x_grid.length; xi++) {
                row.push(flat[yi * x_grid.length + xi]);
            }
            z.push(row);
        }
        return z;
    });
}

function renderDecisionBoundary(z) {
    if (networkPlot && networkPlot.data && networkPlot.data.length > 1 && networkPlot.data[1].type === 'contour') {
        Plotly.restyle(networkPlot, { z: [z] }, [1]);
    } else {
        Plotly.addTraces(networkPlot, [{
            z,
            x: x_grid,
            y: y_grid,
            type: 'contour',
            showscale: false,
            colorscale: [['0', 'rgba(249,115,22,0.8)'], ['1', 'rgba(96,165,250,0.8)']],
            contours: { start: 0.5, end: 0.5, coloring: 'lines' }
        }]);
    }
}

trainBtn.addEventListener('click', async () => {
    if (isTraining) return;
    isTraining = true;
    trainBtn.textContent = "Training...";
    trainBtn.disabled = true;

    // Create and train a real neural network with TensorFlow.js
    const n_neurons = parseInt(neuronsSlider.value);
    const n_layers = hiddenLayers;

    // Prepare the data
    const xs = tf.tensor2d(nn_data.x.map((x, i) => [x, nn_data.y[i]]));
    const ys = tf.tensor2d(nn_data.labels.map(l => [l]));

    // Add progress display
    const progressDiv = document.createElement('div');
    progressDiv.className = 'mt-4 text-sm';
    progressDiv.innerHTML = '<div>Training progress: <span id="train-epoch">0</span>/50 epochs</div>' +
                           '<div>Accuracy: <span id="train-accuracy">-</span></div>' +
                           '<div>Loss: <span id="train-loss">-</span></div>';
    trainBtn.parentNode.appendChild(progressDiv);
    
    const epochSpan = document.getElementById('train-epoch');
    const accuracySpan = document.getElementById('train-accuracy');
    const lossSpan = document.getElementById('train-loss');

    // Build the model
    const model = tf.sequential();

    // Add hidden layers
    for (let i = 0; i < n_layers; i++) {
        model.add(tf.layers.dense({
            units: n_neurons,
            activation: 'relu',
            inputShape: i === 0 ? [2] : undefined
        }));
    }

    // Add output layer
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Fast visualization update (batched predictions + lightweight restyle)
    const updateDecisionBoundary = () => {
        const z = computeDecisionZForModel(model);
        renderDecisionBoundary(z);
    };

    // Train the model with visual feedback
    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        shuffle: true,
        verbose: 0,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Update UI
                epochSpan.textContent = epoch + 1;
                accuracySpan.textContent = (logs.acc !== undefined ? logs.acc : logs.accuracy).toFixed(4);
                lossSpan.textContent = logs.loss.toFixed(4);
                
                // Update visualization every 5 epochs
                if (epoch % 5 === 0 || epoch === 49) {
                    await tf.nextFrame();
                    updateDecisionBoundary();
                }
            }
        }
    });

    // Final update
    updateDecisionBoundary();

    trainBtn.textContent = "Train Network";
    trainBtn.disabled = false;
    isTraining = false;
    progressDiv.remove();

    // Clean up tensors
    xs.dispose();
    ys.dispose();
    model.dispose();
});


updateNetworkBuilderUI();

    // --- 4. ACTIVATION FUNCTION DEMO ---
    const activationSelect = document.getElementById('activation-select');
    const activationPlot = document.getElementById('activation-plot');
    const activationInfo = document.getElementById('activation-info');
    const act_funcs = {
        relu: { func: x => Math.max(0, x), info: "<strong>Most common choice for hidden layers.</strong> Fast and helps prevent vanishing gradients, but can 'die' (always output zero)." },
        leaky_relu: { func: x => x > 0 ? x : 0.1 * x, info: "A fix for the 'dying ReLU' problem. Allows a small, non-zero gradient when the unit is not active." },
        sigmoid: { func: x => 1 / (1 + Math.exp(-x)), info: "Squashes values to a [0, 1] range. <strong>Used for the output layer in binary classification.</strong>" },
        tanh: { func: x => Math.tanh(x), info: "Squashes values to a [-1, 1] range. Zero-centered, but still suffers from vanishing gradients." }
    };
    
    function updateActivationPlot() {
        const selected = activationSelect.value;
        const x = Array.from({length: 101}, (_, i) => (i - 50) / 10);
        const y = x.map(act_funcs[selected].func);
        activationInfo.innerHTML = act_funcs[selected].info;
        Plotly.newPlot(activationPlot, [{ x, y, type:'scatter' }], { title: selected.replace('_',' ') + ' Function', xaxis:{zeroline:true}, yaxis:{zeroline:true}, margin:{t:40,r:10,b:20,l:20} });
    }
    activationSelect.addEventListener('change', updateActivationPlot);
    updateActivationPlot();
});
</script>