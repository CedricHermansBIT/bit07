---
layout: base.njk
title: "Chapter 6: Naïve Bayes Classifiers"
---

<!-- Header -->
<div class="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-2xl p-6 mb-8">
    <h2 class="text-2xl font-bold text-gray-800 mb-2">Chapter 6: Naïve Bayes Classifiers</h2>
    <p class="text-gray-700 leading-relaxed">The classifiers we've seen so far, like Logistic Regression, learn to find a decision boundary that separates classes. In this chapter, we introduce a different family of models: <strong>generative classifiers</strong>, exemplified by the Naïve Bayes algorithm.</p>
</div>

<!-- 1. Generative vs. Discriminative -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">A Different Approach: Generative vs. Discriminative Models</h3>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h4 class="font-semibold text-blue-800">Discriminative Models (e.g., Logistic Regression)</h4>
            <p class="text-sm text-blue-700">These models learn the conditional probability `P(y | X)`. Their primary focus is to find the <strong>decision boundary</strong> between classes.</p>
            <p class="text-sm mt-2"><strong>Analogy:</strong> A customs officer who only learns the rules to distinguish "allowed" from "forbidden" items, without knowing what makes an item belong to either category.</p>
        </div>
        <div class="bg-green-50 p-4 rounded-lg border border-green-200">
            <h4 class="font-semibold text-green-800">Generative Models (e.g., Naïve Bayes)</h4>
            <p class="text-sm text-green-700">These models learn the joint probability distribution `P(X, y)`. They learn what the data for <strong>each class looks like</strong>.</p>
            <p class="text-sm mt-2"><strong>Analogy:</strong> A biologist who studies lions and tigers so thoroughly that they can not only distinguish between them but could also, in theory, generate a new, plausible example of a lion or a tiger.</p>
        </div>
    </div>
</div>

<!-- 2. Bayes' Theorem -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">The Engine: Bayes' Theorem</h3>
    <p class="text-gray-700 mb-4">Generative classifiers use Bayes' Theorem to update our beliefs in light of new evidence.</p>
    <div class="code-block text-center text-lg mb-4">P(Hypothesis | Evidence) = [ P(Evidence | Hypothesis) * P(Hypothesis) ] / P(Evidence)</div>
    <p class="text-gray-700 mb-4>
        Every part of this equation has a specific meaning in the context of classification:
        <ul class="list-disc list-inside mt-2 space-y-1">
            <li><strong>Posterior:</strong> `P(Hypothesis | Evidence)`
    - The updated probability of the hypothesis (class) given the evidence (features).</li>
            <li><strong>Likelihood:</strong> `P(Evidence | Hypothesis)`
    - The probability of observing the evidence if the hypothesis is true.</li>
            <li><strong>Prior:</strong> `P(Hypothesis)`
    - The initial probability of the hypothesis before seeing the evidence.</li>
            <li><strong>Evidence (or Marginal):</strong> `P(Evidence)`
    - The total probability of observing the evidence under all hypotheses.</li>
        </ul>
    </p>    
    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: Is this sequence a promoter?</h4>
    <p class="text-gray-700 mb-4">Let's see how finding a "TATA box" (evidence) in a DNA sequence updates our belief that the sequence is a promoter (hypothesis). Adjust the sliders to see the calculation change in real-time.</p>
    <div class="interactive-demo grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        <div class="space-y-4">
            <div>
                <label class="block text-sm font-medium">Prior: P(Promoter)</label>
                <p class="text-xs text-gray-500 -mt-1 mb-1">The overall chance a random sequence is a promoter, before seeing any evidence.</p>
                <input id="prior-slider" type="range" min="0.01" max="0.5" step="0.01" value="0.01" class="parameter-slider">
                <span id="prior-value" class="text-sm">1%</span>
            </div>
            <div>
                <label class="block text-sm font-medium">Likelihood: P(TATA | Promoter)</label>
                 <p class="text-xs text-gray-500 -mt-1 mb-1">If a sequence IS a promoter, the chance it contains a TATA box.</p>
                <input id="like-promo-slider" type="range" min="0.1" max="1" step="0.05" value="0.8" class="parameter-slider">
                <span id="like-promo-value" class="text-sm">80%</span>
            </div>
            <div>
                <label class="block text-sm font-medium">Likelihood: P(TATA | Not Promoter)</label>
                <p class="text-xs text-gray-500 -mt-1 mb-1">If a sequence IS NOT a promoter, the chance it has a TATA box by random chance.</p>
                <input id="like-not-promo-slider" type="range" min="0.01" max="0.5" step="0.01" value="0.1" class="parameter-slider">
                <span id="like-not-promo-value" class="text-sm">10%</span>
            </div>
        </div>
        <div class="bg-white p-4 rounded-lg">
            <div id="bayes-calculation-steps" class="text-sm font-mono text-left mb-4 p-3 bg-gray-50 rounded">
                <!-- JS will populate this -->
            </div>
            <div class="text-center">
                <div class="text-sm text-gray-600">Final Posterior Probability</div>
                <div class="text-lg font-bold">P(Promoter | TATA box found)</div>
                <div id="posterior-prob" class="text-4xl font-bold font-mono text-primary-600 my-2">7.5%</div>
                <p id="bayes-conclusion" class="text-sm text-gray-700"></p>
            </div>
        </div>
    </div>
</div>

<!-- 3. The Naive Assumption & The Zero-Frequency Problem -->
<div class="card mb-8">
    <h3 class="text-2xl font-bold text-gray-800 mb-4">The "Naïve" Assumption and Its Consequences</h3>
    <p class="text-gray-700 mb-4">Calculating `P(features | class)` directly is often impossible. Naïve Bayes simplifies this by making a strong (or "naïve") assumption: all features are conditionally independent given the class. This allows us to multiply the individual likelihoods:</p>
    <div class="code-block text-center">P(X | y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)</div>
    <div class="highlight mt-4">This assumption is almost always false in bioinformatics (e.g., expression levels of genes in the same pathway are not independent), but the model often performs surprisingly well anyway.</div>

    <h4 class="text-xl font-bold text-gray-800 mt-6 mb-4">Interactive Demo: The Zero-Frequency Problem</h4>
    <p class="text-gray-700 mb-4">A major practical issue arises when a feature value never appears with a particular class in the training data. Because we are multiplying probabilities, a single zero will make the entire posterior probability for that class zero. Let's see this in action.</p>
    <div class="interactive-demo">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <!-- Training Data -->
            <div class="text-sm">
                <h5 class="font-bold text-center mb-2">Training Data (10 sequences)</h5>
                <p><strong>Priors:</strong> P(Gene)=0.5, P(Not Gene)=0.5</p>
                <p class="font-bold mt-2">Likelihoods for "Not Gene":</p>
                <ul class="list-disc list-inside">
                    <li>P(Start Codon=True | Not Gene) = 0/5 = <span class="text-red-600 font-bold">0.0</span></li>
                    <li>P(PolyA=True | Not Gene) = 2/5 = 0.4</li>
                </ul>
            </div>
            <!-- Calculation -->
            <div id="zero-freq-calc" class="lg:col-span-2 p-4 bg-white rounded-lg">
                <!-- JS will populate this -->
            </div>
        </div>
        <div class="text-center mt-4">
            <button id="toggle-smoothing-btn" class="btn-primary">✅ Enable Laplacian Smoothing (α=1)</button>
        </div>
    </div>
</div>

<!-- Numerical underflow and log probabilities -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">Numerical Underflow and Log Probabilities</h3>
    <p class="text-gray-700 mb-4">When dealing with many features, multiplying many small probabilities can lead to numerical underflow, where the result is too small for the computer to represent. To avoid this, we can use logarithms to convert products into sums:</p>
    <div class="code-block text-center">
P(X | y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y) <br>
🔻 <br>
log(P(X | y)) = log(P(x₁|y)) + log(P(x₂|y)) + ... + log(P(xₙ|y))</div>
    <p class="text-gray-700 mt-4">This transformation not only prevents underflow but also simplifies the computation, as addition is computationally cheaper than multiplication.</p>
    <!-- Example -->
    <div class="bg-gray-50 p-4 rounded-lg mt-4">
        <h4 class="font-bold mb-2">Example:</h4>
        <p class="text-sm mb-2">Suppose we have three features with probabilities:</p>
        <ul class="list-disc list-inside text-sm mb-2">
            <li>P(x₁|y) = 0.01</li>
            <li>P(x₂|y) = 0.02</li>
            <li>P(x₃|y) = 0.03</li>
        </ul>
        <p class="text-sm mb-2">Calculating the product directly:</p>
        <div class="code-block text-center mb-2">P(X|y) = 0.01 * 0.02 * 0.03 = 0.000006</div>
        <p class="text-sm mb-2">Using logarithms:</p>
        <div class="code-block text-center">log(P(X|y)) = log(0.01) + log(0.02) + log(0.03) <br>
= -4.605 + -3.912 + -3.507 = -12.024</div>
<p class="text-sm mb-2">Note that we are not interested in the actual number, just in the proportions to one another.</p>
</div>
</div>

<!-- 4. Types of Naive Bayes & Implementation -->
<div class="card mb-8">
    <h3 class="text-xl font-bold text-gray-800 mb-4">Types of Naïve Bayes & Scikit-Learn Implementation</h3>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 text-sm">
        <div class="bg-gray-100 p-3 rounded-lg">
            <h4 class="font-semibold">GaussianNB</h4>
            <p>For continuous features that are assumed to follow a normal (Gaussian) distribution.</p>
        </div>
        <div class="bg-gray-100 p-3 rounded-lg">
            <h4 class="font-semibold">MultinomialNB</h4>
            <p>For discrete features (e.g., word counts, k-mer counts). Excellent for text or sequence classification.</p>
        </div>
        <div class="bg-gray-100 p-3 rounded-lg">
            <h4 class="font-semibold">BernoulliNB</h4>
            <p>For binary features (e.g., presence/absence of a mutation).</p>
        </div>
    </div>
    <p class="text-gray-700 mb-4">Here is how you would implement a classifier for DNA dinucleotide frequencies. Notice the `alpha=1.0` parameter, which enables Laplacian smoothing by default.</p>
    <div class="code-block">
<pre><code class="language-python">from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# X_counts: matrix of dinucleotide counts for each sequence
# y_species: the species label for each sequence
X_train, X_test, y_train, y_test = train_test_split(X_counts, y_species)
# 1. Create the model with Laplacian smoothing (alpha=1.0)
model = MultinomialNB(alpha=1.0)
# 2. Train the model
model.fit(X_train, y_train)
# 3. Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)</code></pre>
</div>
</div>

<script>
document.addEventListener('DOMContentLoaded', () => {
    // --- 2. BAYES THEOREM DEMO ---
    const priorSlider = document.getElementById('prior-slider');
    const likePromoSlider = document.getElementById('like-promo-slider');
    const likeNotPromoSlider = document.getElementById('like-not-promo-slider');
    const bayesCalcDiv = document.getElementById('bayes-calculation-steps');

    function updateBayesCalc() {
        // Get values
        const pPromo = parseFloat(priorSlider.value);
        const pNotPromo = 1 - pPromo;
        const pTataGivenPromo = parseFloat(likePromoSlider.value);
        const pTataGivenNotPromo = parseFloat(likeNotPromoSlider.value);

        // Update slider value displays
        document.getElementById('prior-value').textContent = `${(pPromo * 100).toFixed(0)}%`;
        document.getElementById('like-promo-value').textContent = `${(pTataGivenPromo * 100).toFixed(0)}%`;
        document.getElementById('like-not-promo-value').textContent = `${(pTataGivenNotPromo * 100).toFixed(0)}%`;

        // --- Perform Calculations ---
        // 1. Calculate P(TATA) - the evidence
        const pTata = (pTataGivenPromo * pPromo) + (pTataGivenNotPromo * pNotPromo);
        
        // 2. Calculate P(Promo|TATA) - the posterior
        const numerator = pTataGivenPromo * pPromo;
        const pPromoGivenTata = numerator / pTata;

        // --- Update Calculation Steps Display ---
        bayesCalcDiv.innerHTML = `
            <p class="font-bold border-b pb-1 mb-1">Calculation Breakdown:</p>
            <p>1. Find P(TATA) (the evidence):</p>
            <p class="pl-2 text-gray-600">P(TATA|P)*P(P) + P(TATA|~P)*P(~P)</p>
            <p class="pl-2">= (${pTataGivenPromo.toFixed(2)} * ${pPromo.toFixed(2)}) + (${pTataGivenNotPromo.toFixed(2)} * ${pNotPromo.toFixed(2)})</p>
            <p class="pl-2">= <span class="font-bold text-blue-600">${pTata.toFixed(4)}</span></p>
            <p class="mt-2">2. Calculate Posterior:</p>
            <p class="pl-2 text-gray-600">(P(TATA|P) * P(P)) / P(TATA)</p>
            <p class="pl-2">= (${pTataGivenPromo.toFixed(2)} * ${pPromo.toFixed(2)}) / ${pTata.toFixed(4)}</p>
            <p class="pl-2">= ${numerator.toFixed(4)} / ${pTata.toFixed(4)} = <span class="font-bold text-green-600">${pPromoGivenTata.toFixed(4)}</span></p>
        `;

        // --- Update Final Result ---
        document.getElementById('posterior-prob').textContent = `${(pPromoGivenTata * 100).toFixed(1)}%`;
        document.getElementById('bayes-conclusion').textContent = `Finding a TATA box updated our belief from a prior of ${(pPromo * 100).toFixed(1)}% to a posterior of ${(pPromoGivenTata * 100).toFixed(1)}%.`;
    }
    [priorSlider, likePromoSlider, likeNotPromoSlider].forEach(s => s.addEventListener('input', updateBayesCalc));
    updateBayesCalc();


    // --- 3. ZERO FREQUENCY DEMO ---
    const toggleSmoothingBtn = document.getElementById('toggle-smoothing-btn');
    const calcDiv = document.getElementById('zero-freq-calc');
    let smoothingEnabled = false;

    function updateZeroFreqCalc() {
        if (smoothingEnabled) {
            toggleSmoothingBtn.textContent = '❌ Disable Laplacian Smoothing';
            toggleSmoothingBtn.classList.remove('btn-primary');
            toggleSmoothingBtn.classList.add('btn-secondary');
            calcDiv.innerHTML = `
                <h5 class="font-bold text-center mb-2">Classify: (Start Codon=True, PolyA=True)</h5>
                <p class="font-semibold text-green-700">With Smoothing (α=1)</p>
                <div class="text-sm font-mono mt-2">
                    <p>P(Not Gene | X) ∝ P(Not Gene) * P(Start=T|Not Gene) * P(PolyA=T|Not Gene)</p>
                    <p class="ml-4">= 0.5 * <span class="bg-green-200 p-1 rounded">(0+1)/(5+1*2)</span> * (2+1)/(5+1*2)</p>
                    <p class="ml-4">= 0.5 * 0.143 * 0.429 = <span class="font-bold">0.0307</span></p>
                </div>
                <div class="text-sm font-mono mt-2">
                    <p>P(Gene | X) ∝ P(Gene) * P(Start=T|Gene) * P(PolyA=T|Gene)</p>
                    <p class="ml-4">= 0.5 * (4/5) * (2/5) = <span class="font-bold">0.16</span></p>
                </div>
                <p class="text-center mt-3 font-bold text-lg text-green-700">Verdict: Classify as "Gene"</p>
            `;
        } else {
            toggleSmoothingBtn.textContent = '✅ Enable Laplacian Smoothing (α=1)';
            toggleSmoothingBtn.classList.remove('btn-secondary');
            toggleSmoothingBtn.classList.add('btn-primary');
            calcDiv.innerHTML = `
                <h5 class="font-bold text-center mb-2">Classify: (Start Codon=True, PolyA=True)</h5>
                <p class="font-semibold text-red-700">Without Smoothing</p>
                <div class="text-sm font-mono mt-2">
                    <p>P(Not Gene | X) ∝ P(Not Gene) * P(Start=T|Not Gene) * P(PolyA=T|Not Gene)</p>
                    <p class="ml-4">= 0.5 * <span class="bg-red-200 p-1 rounded">0.0</span> * 0.4 = <span class="font-bold text-red-600">0.0</span></p>
                </div>
                <div class="text-sm font-mono mt-2">
                    <p>P(Gene | X) ∝ P(Gene) * P(Start=T|Gene) * P(PolyA=T|Gene)</p>
                    <p class="ml-4">= 0.5 * (4/5) * (2/5) = <span class="font-bold">0.16</span></p>
                </div>
                <p class="text-center mt-3 font-bold text-lg text-red-700">Verdict: Impossible to be "Not Gene"</p>
            `;
        }
    }

    toggleSmoothingBtn.addEventListener('click', () => {
        smoothingEnabled = !smoothingEnabled;
        updateZeroFreqCalc();
    });
    updateZeroFreqCalc();
});
</script>