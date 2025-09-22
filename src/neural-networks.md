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

<!-- 5. Advanced Architectures -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">Beyond Feedforward: A Glimpse into Advanced Architectures</h3>
    <p class="text-gray-700 mb-4">While feedforward networks are powerful, specialized architectures have revolutionized bioinformatics:</p>
    <div class="space-y-4">
        <div class="bg-gray-50 p-3 rounded-lg">
            <h4 class="font-semibold">Convolutional Neural Networks (CNNs)</h4>
            <p class="text-sm">The state-of-the-art for grid-like data. In bioinformatics, they are used to scan DNA/RNA sequences for motifs or classify cell types in microscopy images.</p>
        </div>
        <div class="bg-gray-50 p-3 rounded-lg">
            <h4 class="font-semibold">Recurrent Neural Networks (RNNs)</h4>
            <p class="text-sm">Designed for sequential data. Used to predict protein secondary structure from the primary amino acid sequence.</p>
        </div>
        <div class="bg-gray-50 p-3 rounded-lg">
            <h4 class="font-semibold">Transformers</h4>
            <p class="text-sm">The current frontier. The groundbreaking <strong>AlphaFold2</strong> model uses a Transformer-based architecture to predict protein 3D structures with incredible accuracy.</p>
        </div>
    </div>
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