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
    return parts[0];
  }

  // Curated examples per chapter (Python-focused), adapted from BIT07.txt
  const EXAMPLES = {
    'linear-regression': [
      {
        title: 'Train/Test Split and LinearRegression',
        language: 'python',
        code: `import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\n\n# Toy data\ndf = pd.DataFrame({\n    'tf_abundance': np.random.rand(200) * 10,\n    'gene_expression': 2.5 + 3 * (np.random.rand(200) * 10) + np.random.randn(200) * 2\n})\nX = df[['tf_abundance']]\ny = df['gene_expression']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodel = LinearRegression().fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint('Intercept:', model.intercept_)\nprint('Slope:', model.coef_[0])\nprint('MSE:', mean_squared_error(y_test, y_pred))\nprint('R2:', r2_score(y_test, y_pred))`
      },
      {
        title: 'Regularization: Ridge and Lasso',
        language: 'python',
        code: `from sklearn.linear_model import Ridge, Lasso\n\nridge = Ridge(alpha=1.0).fit(X_train, y_train)\nlasso = Lasso(alpha=1.0).fit(X_train, y_train)\nprint('Ridge R2:', ridge.score(X_test, y_test))\nprint('Lasso R2:', lasso.score(X_test, y_test))\nprint('Non-zero Lasso features:', (lasso.coef_ != 0).sum())`
      }
    ],
    'logistic-regression': [
      {
        title: 'Scaling + Logistic Regression',
        language: 'python',
        code: `import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, roc_auc_score\n\n# Example data\ndf = pd.DataFrame({\n    'expression': [1.2, 0.7, 2.4, 3.3, 0.2, 1.8, 2.1, 0.5],\n    'ppi_count': [3, 1, 8, 10, 0, 4, 6, 2],\n    'is_essential': [0, 0, 1, 1, 0, 1, 1, 0]\n})\nX = df[['expression', 'ppi_count']]\ny = df['is_essential']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n\nscaler = StandardScaler().fit(X_train)\nX_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)\n\nmodel = LogisticRegression(C=1.0, random_state=42).fit(X_train_s, y_train)\ny_pred = model.predict(X_test_s)\nproba = model.predict_proba(X_test_s)[:,1]\nprint(classification_report(y_test, y_pred))\nprint('AUC:', roc_auc_score(y_test, proba))`
      },
      {
        title: 'Polynomial Features (non-linear boundary)',
        language: 'python',
        code: `from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree=2, include_bias=False)\nX_train_poly = poly.fit_transform(X_train_s)\nX_test_poly = poly.transform(X_test_s)\npoly_model = LogisticRegression(max_iter=2000).fit(X_train_poly, y_train)\nprint('Score with poly:', poly_model.score(X_test_poly, y_test))`
      }
    ],
    'hyperparameter-tuning': [
      {
        title: 'GridSearchCV (Logistic Regression)',
        language: 'python',
        code: `from sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\n\nX, y = make_classification(n_samples=1000, n_features=20, random_state=42)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodel = LogisticRegression(solver='liblinear')\nparam_grid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}\nsearch = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)\nsearch.fit(X_train, y_train)\nprint('Best params:', search.best_params_)\nprint('CV best score:', search.best_score_)\nprint('Test accuracy:', search.score(X_test, y_test))`
      },
      {
        title: 'RandomizedSearchCV',
        language: 'python',
        code: `from sklearn.model_selection import RandomizedSearchCV\nfrom scipy.stats import uniform\n\nparam_dist = {'penalty': ['l1', 'l2'], 'C': uniform(0.01, 99.99)}\nrand = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)\nrand.fit(X_train, y_train)\nprint('Best params:', rand.best_params_)\nprint('CV best score:', rand.best_score_)`
      }
    ],
    'imbalanced-data': [
      {
        title: 'SMOTE Oversampling',
        language: 'python',
        code: `from imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report\n\n# X, y assumed defined\nX_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)\nX_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)\nclf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X_res, y_res)\nprint(classification_report(y_test, clf.predict(X_test)))`
      },
      {
        title: 'Tune with better scorer',
        language: 'python',
        code: `from sklearn.model_selection import GridSearchCV\n# grid and model as before\nsearch = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='f1_weighted')\nsearch.fit(X_train, y_train)\nprint('Best params (F1-weighted):', search.best_params_)`
      }
    ],
    'naive-bayes': [
      {
        title: 'MultinomialNB with Laplace smoothing',
        language: 'python',
        code: `import numpy as np\nimport pandas as pd\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report\n\nnp.random.seed(42)\nX = pd.DataFrame(np.random.randint(0,100,(500,8)), columns=list('ABCDEFGH'))\ny = (X['C'] + X['G'] > 100).astype(int)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)\nnb = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)\nprint(classification_report(y_te, nb.predict(X_te)))\nprint('Proba of first sample:', nb.predict_proba(X_te.iloc[[0]]))`
      }
    ],
    'decision-trees': [
      {
        title: 'DecisionTreeClassifier (pre-pruned)',
        language: 'python',
        code: `from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import classification_report\n# X_train, X_test, y_train, y_test assumed defined\nclf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)\ny_pred = clf.predict(X_test)\nprint(classification_report(y_test, y_pred))`
      },
      {
        title: 'Gini vs Entropy',
        language: 'python',
        code: `gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42).fit(X_train, y_train)\nentropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42).fit(X_train, y_train)\nprint('Gini score:', gini_tree.score(X_test, y_test))\nprint('Entropy score:', entropy_tree.score(X_test, y_test))`
      }
    ],
    'ensemble-learning': [
      {
        title: 'BaggingClassifier with LogisticRegression',
        language: 'python',
        code: `from sklearn.ensemble import BaggingClassifier\nfrom sklearn.linear_model import LogisticRegression\nbase = LogisticRegression(solver='liblinear')\nbag = BaggingClassifier(base_estimator=base, n_estimators=50, n_jobs=-1, random_state=42).fit(X_train, y_train)\nprint('Bagging score:', bag.score(X_test, y_test))`
      },
      {
        title: 'AdaBoost with Decision Stumps',
        language: 'python',
        code: `from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nbase = DecisionTreeClassifier(max_depth=1, random_state=42)\nada = AdaBoostClassifier(base_estimator=base, n_estimators=100, learning_rate=0.5, random_state=42).fit(X_train, y_train)\nprint('AdaBoost score:', ada.score(X_test, y_test))`
      }
    ],
    'clustering': [
      {
        title: 'K-Means + SSE (elbow)',
        language: 'python',
        code: `from sklearn.datasets import make_blobs\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nimport numpy as np\n\nX, _ = make_blobs(n_samples=300, centers=3, random_state=42)\nsse, sil = [], []\nfor k in range(2, 9):\n    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)\n    sse.append(km.inertia_)\n    sil.append(silhouette_score(X, km.labels_))\nprint('SSE:', sse)\nprint('Silhouette:', np.round(sil, 3))`
      },
        {
            title: 'Hierarchical Clustering',
            language: 'python',
            code: `from sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import silhouette_score\nimport numpy as np\n\nX, _ = make_blobs(n_samples=300, centers=3, random_state=42)\n\n# AgglomerativeClustering\nagg = AgglomerativeClustering(n_clusters=3, linkage='ward', random_state=42)\nagg.fit(X)\nprint('Agglomerative score:', agg.score(X))\n\n# Hierarchical clustering\nhclust = HierarchicalClustering(n_clusters=3, linkage='ward')\nhclust.fit(X)\nprint('Hierarchical score:', hclust.score(X))\n\n# Silhouette score\nsil = silhouette_score(X, agg.labels_, metric='euclidean')\nprint('Silhouette:', np.round(sil, 3))`
        }
    ],
    'dimensionality-reduction': [
      {
        title: 'PCA (2 components)',
        language: 'python',
        code: `from sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nimport numpy as np\n# X assumed defined (n_samples, n_features)\nX_s = StandardScaler().fit_transform(X)\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_s)\nprint('Explained variance ratio:', np.round(pca.explained_variance_ratio_, 3))`
      }
    ],
    'ml-workflows': [
      {
        title: 'Pipeline + ColumnTransformer',
        language: 'python',
        code: `from sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.ensemble import RandomForestClassifier\n\nnumerical = ['gene_expression_level', 'age']\ncategorical = ['tumor_stage', 'treatment_group']\n\npre = ColumnTransformer([\n  ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numerical),\n  ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), categorical)\n])\n\npipe = Pipeline([('pre', pre), ('clf', RandomForestClassifier(random_state=42))])\npipe.fit(X_train, y_train)\nprint('Accuracy:', pipe.score(X_test, y_test))`
      },
      {
        title: 'FeatureUnion pattern',
        language: 'python',
        code: `from sklearn.pipeline import FeatureUnion\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.base import BaseEstimator, TransformerMixin\n\nclass ColumnSelector(BaseEstimator, TransformerMixin):\n    def __init__(self, columns): self.columns = columns\n    def fit(self, X, y=None): return self\n    def transform(self, X): return X[self.columns]\n\nunion = FeatureUnion([\n  ('pca', Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=10))])),\n  ('biomarkers', Pipeline([('sel', ColumnSelector(columns=['gene_0','gene_1','gene_2','gene_3','gene_4']))]))\n])\nX_trf = union.fit_transform(X_train)\nprint('Shape:', X_trf.shape)`
      }
    ],
    'neural-networks': [
      {
        title: 'Keras Sequential (binary classification)',
        language: 'python',
        code: `import tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense\nfrom sklearn.model_selection import train_test_split\nimport numpy as np\n\nX = np.random.rand(1000, 100)\ny = np.random.randint(0, 2, 1000)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodel = Sequential([\n    Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),\n    Dense(32, activation='relu'),\n    Dense(1, activation='sigmoid')\n])\nmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\nmodel.fit(X_tr, y_tr, epochs=10, batch_size=32, validation_split=0.1, verbose=0)\nprint('Test acc:', model.evaluate(X_te, y_te, verbose=0)[1])`
      }
    ],
    'unsupervised-learning': [
      {
        title: 'K-Means + Silhouette',
        language: 'python',
        code: `from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n# X assumed defined\nkm = KMeans(n_clusters=3, n_init='auto', random_state=42).fit(X)\nprint('Silhouette:', silhouette_score(X, km.labels_))`
      }
    ],
    'introduction': [
      {
        title: 'Project skeleton: train/test split',
        language: 'python',
        code: `from sklearn.model_selection import train_test_split\n# X, y = ... load your data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint(X_train.shape, X_test.shape)`
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
