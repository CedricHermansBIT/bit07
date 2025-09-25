// Shared interactive behavior and chapter code examples injector
(function() {
  const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';
  if (!isBrowser) return;

  // Utility: make code block HTML
  function sectionHtml(sec, idx) {
    const langClass = sec.language ? `language-${sec.language}` : '';
    const descHtml = sec.description ? `<p class="text-sm text-gray-600 mb-2">${sec.description}</p>` : '';
    return `
      <section class="card">
        <div class="flex items-start justify-between gap-4 mb-3">
          <h3 class="text-base md:text-lg font-semibold text-gray-800">${sec.title || 'Example ' + (idx+1)}</h3>
          ${sec.badge ? `<span class="text-xs px-2 py-1 rounded bg-primary-50 text-primary-700 border border-primary-200">${sec.badge}</span>` : ''}
        </div>
        ${descHtml}
        <div class="relative">
          <button class="copy-btn absolute top-2 right-2 px-2 py-1 bg-gray-700 hover:bg-gray-600 text-gray-100 text-xs rounded transition-colors">Copy</button>
          <pre class="rounded-lg overflow-x-auto"><code class="${langClass}">${escapeHtml(sec.code)}</code></pre>
        </div>
      </section>`;
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;');
  }

  function highlightNewCode(container) {
    if (!window.hljs) return;
    container.querySelectorAll('pre code').forEach((el) => {
      try { window.hljs.highlightElement(el); } catch (_) {}
    });
  }

  function attachCopyHandlers(container) {
    container.querySelectorAll('button.copy-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const code = btn.parentElement.querySelector('code').innerText;
        navigator.clipboard.writeText(code);
        const prev = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = prev, 1500);
      });
    });
  }

  function chapterKeyFromPath(pathname) {
    const prefix = (window.__PATH_PREFIX__ || '/');
    // Ensure prefix ends with '/'
    const normPrefix = prefix.endsWith('/') ? prefix : prefix + '/';
    let path = pathname || '/';
    // Strip the prefix if present
    if (path.startsWith(normPrefix)) {
      path = path.slice(normPrefix.length - 1); // keep leading '/'
    }
    const parts = (path || '/').split('/').filter(Boolean);
    // Expect structure /<chapter>/ or /
    if (parts.length === 0) return 'home';
    if (parts.length === 1) return parts[0];
    return parts[1];
  }

  // Curated examples per chapter (Python-focused), adapted from BIT07.txt
  const EXAMPLES = {
'linear-regression': [
  {
    title: 'Train/Test Split and LinearRegression',
    language: 'python',
    code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic data for a simple regression problem
np.random.seed(42)
df = pd.DataFrame({
    'tf_abundance': np.random.rand(200) * 10,
    'gene_expression': 2.5 + 3 * (np.random.rand(200) * 10) + np.random.randn(200) * 2
})
X = df[['tf_abundance']]
y = df['gene_expression']

# 2. Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a linear regression model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Evaluate the model
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")`
  },
  {
    title: 'Regularization: Ridge and Lasso',
    language: 'python',
    code: `from sklearn.linear_model import Ridge, Lasso

# Train a Ridge model (L2 regularization)
# alpha controls the strength of regularization
ridge = Ridge(alpha=1.0).fit(X_train, y_train)

# Train a Lasso model (L1 regularization)
lasso = Lasso(alpha=1.0).fit(X_train, y_train)

print(f"Ridge R2: {ridge.score(X_test, y_test):.2f}")
print(f"Lasso R2: {lasso.score(X_test, y_test):.2f}")
# Lasso can drive some feature coefficients to exactly zero
print(f"Non-zero Lasso features: {(lasso.coef_ != 0).sum()}")`
  }
],
'logistic-regression': [
  {
    title: 'Scaling + Logistic Regression',
    language: 'python',
    code: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Example data for binary classification
df = pd.DataFrame({
    'expression': [1.2, 0.7, 2.4, 3.3, 0.2, 1.8, 2.1, 0.5],
    'ppi_count': [3, 1, 8, 10, 0, 4, 6, 2],
    'is_essential': [0, 0, 1, 1, 0, 1, 1, 0]
})
X = df[['expression', 'ppi_count']]
y = df['is_essential']

# Stratify ensures the train/test split has the same class proportions as the original dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features to have zero mean and unit variance
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(C=1.0, random_state=42).fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, proba):.2f}")`
  },
  {
    title: 'Polynomial Features (non-linear boundary)',
    language: 'python',
    code: `from sklearn.preprocessing import PolynomialFeatures

# Create interaction and polynomial features to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_s)
X_test_poly = poly.transform(X_test_s)

# Train a new model on the polynomial features
poly_model = LogisticRegression(max_iter=2000).fit(X_train_poly, y_train)
print(f"Score with polynomial features: {poly_model.score(X_test_poly, y_test):.2f}")`
  }
],
'hyperparameter-tuning': [
  {
    title: 'GridSearchCV (Logistic Regression)',
    language: 'python',
    code: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear')

# Define the grid of hyperparameters to search
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100]
}

# Set up the grid search with 5-fold cross-validation
search = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1 # Use all available CPU cores
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"CV best score: {search.best_score_:.3f}")
print(f"Test accuracy: {search.score(X_test, y_test):.3f}")`
  },
  {
    title: 'RandomizedSearchCV',
    language: 'python',
    code: `from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define hyperparameter distributions to sample from
param_dist = {
    'penalty': ['l1', 'l2'],
    'C': uniform(loc=0.01, scale=99.99) # Sample C from a uniform distribution
}

# Set up the randomized search
rand = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50, # Number of parameter settings to sample
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
rand.fit(X_train, y_train)

print(f"Best params: {rand.best_params_}")
print(f"CV best score: {rand.best_score_:.3f}")`
  }
],
'imbalanced-data': [
  {
    title: 'SMOTE Oversampling',
    language: 'python',
    code: `from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# X, y assumed defined from a dataset with imbalanced classes
# Use SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train a classifier on the balanced data
# 'class_weight="balanced"' also helps by adjusting weights inversely proportional to class frequencies
clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_res, y_res)

print(classification_report(y_test, clf.predict(X_test)))`
  },
  {
    title: 'Tune with a better scorer',
    language: 'python',
    code: `from sklearn.model_selection import GridSearchCV

# For imbalanced data, 'accuracy' is a poor metric.
# 'f1_weighted' is often better as it balances precision and recall.
search = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted'
)
search.fit(X_train, y_train)

print(f"Best params (F1-weighted): {search.best_params_}")`
  }
],
'naive-bayes': [
  {
    title: 'MultinomialNB with Laplace smoothing',
    language: 'python',
    code: `import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate synthetic count data
np.random.seed(42)
X = pd.DataFrame(np.random.randint(0, 100, (500, 8)), columns=list('ABCDEFGH'))
y = (X['C'] + X['G'] > 100).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# alpha=1.0 applies Laplace smoothing to handle features not seen in training data
nb = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)

print(classification_report(y_te, nb.predict(X_te)))
print('Proba of first sample:', nb.predict_proba(X_te.iloc[[0]]))`
  }
],
'decision-trees': [
  {
    title: 'DecisionTreeClassifier (pre-pruned)',
    language: 'python',
    code: `from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# X_train, X_test, y_train, y_test assumed defined
# Pre-prune the tree by limiting its max depth to prevent overfitting
clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))`
  },
  {
    title: 'Gini vs Entropy',
    language: 'python',
    code: `# Train a tree using the 'gini' impurity criterion
gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
gini_tree.fit(X_train, y_train)

# Train a tree using the 'entropy' (information gain) criterion
entropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
entropy_tree.fit(X_train, y_train)

print(f"Gini score: {gini_tree.score(X_test, y_test):.3f}")
print(f"Entropy score: {entropy_tree.score(X_test, y_test):.3f}")`
  }
],
'ensemble-learning': [
  {
    title: 'BaggingClassifier with LogisticRegression',
    language: 'python',
    code: `from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

# Use Logistic Regression as the base estimator
base = LogisticRegression(solver='liblinear')

# Create a bagging ensemble of 50 logistic regression models
bag = BaggingClassifier(
    base_estimator=base,
    n_estimators=50,
    n_jobs=-1,
    random_state=42
).fit(X_train, y_train)

print(f"Bagging score: {bag.score(X_test, y_test):.3f}")`
  },
  {
    title: 'AdaBoost with Decision Stumps',
    language: 'python',
    code: `from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Use a 'decision stump' (a tree with max_depth=1) as the weak learner
base = DecisionTreeClassifier(max_depth=1, random_state=42)

# Create an AdaBoost ensemble
ada = AdaBoostClassifier(
    base_estimator=base,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
).fit(X_train, y_train)

print(f"AdaBoost score: {ada.score(X_test, y_test):.3f}")`
  }
],
'clustering': [
  {
    title: 'K-Means + SSE (elbow)',
    language: 'python',
    code: `from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
sse, sil = [], []

# Test k from 2 to 8 to find the optimal number of clusters
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)
    sse.append(km.inertia_) # Sum of Squared Errors (SSE)
    sil.append(silhouette_score(X, km.labels_))

# The 'elbow' in the SSE plot can suggest the best k
print('SSE:', [round(s, 2) for s in sse])
# Silhouette score closer to 1 is better
print('Silhouette:', np.round(sil, 3))`
  },
  {
    title: 'Hierarchical Clustering',
    language: 'python',
    code: `from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Perform agglomerative clustering with Ward linkage
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)
labels = agg.fit_predict(X)

# Calculate the silhouette score to evaluate cluster quality
sil = silhouette_score(X, labels, metric='euclidean')
print(f"Silhouette Score: {sil:.3f}")`
  }
],
'dimensionality-reduction': [
  {
    title: 'PCA (2 components)',
    language: 'python',
    code: `from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# X assumed defined (n_samples, n_features)
# 1. Scale data before applying PCA
X_s = StandardScaler().fit_transform(X)

# 2. Initialize PCA to reduce to 2 components
pca = PCA(n_components=2)

# 3. Fit and transform the data
X_pca = pca.fit_transform(X_s)

print(f"Shape after PCA: {X_pca.shape}")
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))`
  }
],
'ml-workflows': [
  {
    title: 'Pipeline + ColumnTransformer',
    language: 'python',
    code: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

numerical_features = ['gene_expression_level', 'age']
categorical_features = ['tumor_stage', 'treatment_group']

# Create a preprocessor to handle numerical and categorical data separately
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the full pipeline including the preprocessor and a classifier
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the entire pipeline on the raw data
pipe.fit(X_train, y_train)
print(f"Pipeline accuracy: {pipe.score(X_test, y_test):.3f}")`
  },
  {
    title: 'FeatureUnion pattern',
    language: 'python',
    code: `from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to select specific columns from a DataFrame
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.columns]

# Combine features from different sources: PCA and a direct selection of biomarkers
combined_features = FeatureUnion([
    ('pca', Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10))
    ])),
    ('biomarkers', ColumnSelector(columns=['gene_0', 'gene_1', 'gene_2']))
])

# Fit and transform the data
X_transformed = combined_features.fit_transform(X_train)
print(f"Shape of combined features: {X_transformed.shape}")`
  }
],
'neural-networks': [
  {
    title: 'Keras Sequential (binary classification)',
    language: 'python',
    code: `import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
X = np.random.rand(1000, 100)
y = np.random.randint(0, 2, 1000)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple sequential model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_tr, y_tr,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0 # Suppress verbose output
)

# Evaluate on the test set
loss, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"Test accuracy: {acc:.3f}")`
  }
],
'unsupervised-learning': [
  {
    title: 'K-Means + Silhouette',
    language: 'python',
    code: `from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# X assumed defined
# Fit K-Means with 3 clusters
km = KMeans(
    n_clusters=3,
    n_init='auto', # Run with different centroid seeds
    random_state=42
).fit(X)

# The silhouette score measures how similar an object is to its own cluster
# compared to other clusters. A score closer to 1 is better.
score = silhouette_score(X, km.labels_)
print(f"Silhouette Score: {score:.3f}")`
  }
],
'introduction': [
  {
    title: 'Project skeleton: train/test split',
    language: 'python',
    code: `from sklearn.model_selection import train_test_split

# X, y = ... load your data here

# Split data into training (80%) and testing (20%) sets
# random_state ensures the split is the same every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")`
  }
]
  };

  function renderExamplesForChapter(key) {
    const panel = document.getElementById('code-examples-panel');
    const container = document.getElementById('code-examples-content');
    const btnToggle = document.getElementById('code-examples-toggle');
    if (!panel || !container || !btnToggle) return;

    const examples = EXAMPLES[key] || EXAMPLES[aliasToKey(key)] || null;
    if (!examples || !examples.length) {
      // Hide toggle if no examples
      btnToggle.classList.add('hidden');
      return;
    }

    // Fill content
    container.innerHTML = examples.map(sectionHtml).join('');
    highlightNewCode(container);
    attachCopyHandlers(container);

    // Backdrop and close events
    const backdrop = document.getElementById('code-examples-backdrop');
    const closeBtn = document.getElementById('code-examples-close');
    if (backdrop) backdrop.addEventListener('click', () => panel.classList.add('hidden'));
    if (closeBtn) closeBtn.addEventListener('click', () => panel.classList.add('hidden'));
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') panel.classList.add('hidden');
    });
  }

  function aliasToKey(key) {
    // Map some alternate slugs to primary keys
    const map = {
      'ml-workflows': 'ml-workflows',
      'neural-networks': 'neural-networks',
      'hyperparameter-tuning': 'hyperparameter-tuning',
      'unsupervised-learning': 'unsupervised-learning',
      'dimensionality-reduction': 'dimensionality-reduction'
    };
    return map[key] || key;
  }

  document.addEventListener('DOMContentLoaded', () => {
    const key = chapterKeyFromPath(window.location.pathname);
    renderExamplesForChapter(key);
  });
})();

(function() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('lp') === '1') {
    const navbar = document.getElementById('navbar');
    const footer = document.getElementById('footer');
    const next_steps = document.getElementById('next-steps');
    if (navbar) navbar.style.display = 'none';
    if (footer) footer.style.display = 'none';
    if (next_steps) next_steps.style.display = 'none';
    // set bg-color to white
    const body = document.body;
    body.style.backgroundColor = '#fff';
  }
})();
