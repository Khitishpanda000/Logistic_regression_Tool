import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)

penalty = st.sidebar.selectbox(
    'Regularization',
    ('l2', 'l1', 'elasticnet', 'none')
)

c_input = float(st.sidebar.number_input('C', value=1.0))

solver = st.sidebar.selectbox(
    'Solver',
    ('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga')
)

max_iter = int(st.sidebar.number_input('Max Iterations', value=100))

# Note: scikit-learn dropped multi_class parameter natively, we handle it via wrappers below
multi_class_selection = st.sidebar.selectbox(
    'Multi Class Strategy',
    ('auto/multinomial', 'ovr')
)

# l1_ratio must be a float between 0.0 and 1.0, and only works with elasticnet
l1_ratio = float(st.sidebar.number_input('l1 Ratio (Only for ElasticNet)', min_value=0.0, max_value=1.0, value=0.5, step=0.1))

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    # Safeguard against scikit-learn parameter mismatches
    if penalty == 'elasticnet' and solver != 'saga':
        st.error("❌ 'elasticnet' regularization only works with the 'saga' solver.")
    elif penalty == 'l1' and solver not in ['liblinear', 'saga']:
        st.error(f"❌ 'l1' regularization does not work with the '{solver}' solver. Use 'liblinear' or 'saga'.")
    else:
        # 1. Build base model without the old multi_class parameter
        # Note: 'none' is deprecated in newer versions, pass None instead
        penalty_param = None if penalty == 'none' else penalty
        l1_param = l1_ratio if penalty == 'elasticnet' else None

        base_clf = LogisticRegression(
            penalty=penalty_param,
            C=c_input,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_param
        )

        # 2. Implement the explicit 'ovr' multi-class behavior using OneVsRestClassifier if chosen
        if multi_class_selection == 'ovr':
            clf = OneVsRestClassifier(base_clf)
        else:
            clf = base_clf

        # Fit model and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Draw decision boundary mesh
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)

        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig = st.pyplot(fig)
        
        # Fixed text from "Decision Tree" to "Logistic Regression"
        st.subheader("Accuracy for Logistic Regression: " + str(round(accuracy_score(y_test, y_pred), 2)))
