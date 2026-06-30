# Interactive Logistic Regression Visualizer

An interactive web application built with **Streamlit** to visualize how different hyperparameters impact a **Logistic Regression** classifier's decision boundaries. The tool supports both binary and multiclass synthetic datasets generated using scikit-learn.

---

## 🚀 Features

* **Live Boundary Visualization**: Instantly updates the decision boundary contour mesh on user actions.
* **Dataset Switching**: Toggle cleanly between `Binary` (2 classes) and `Multiclass` (3 classes) datasets.
* **Hyperparameter Tuning**: Real-time experimentation with parameters like Regularization (`l1`, `l2`, `elasticnet`, `none`), Inverse Regularization Strength (`C`), Solvers, and Maximum Iterations.
* **Dynamic Validation Protections**: Interface warnings prevent crashes by validating compatible solver/penalty combinations in real time.
* **Modern UI Elements**: Leverages `fivethirtyeight` plotting aesthetics combined with modern Streamlit controls.

---

## 🛠️ Installation & Setup

Follow these short steps to install the required dependencies and spin up the dashboard locally.

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Install the required scientific stack and deployment libraries via pip:
```bash
pip install streamlit numpy matplotlib scikit-learn
```

### 3. Run the Application
Navigate to your project directory containing `Tool.py` and run the following command:
```bash
streamlit run Tool.py
```

---

## 🎛️ Hyperparameters Explained

| Parameter | UI Control Type | Purpose |
| :--- | :--- | :--- |
| **Select Dataset** | Dropdown | Switches data generation between 2 or 3 distinct isotropic Gaussian blobs. |
| **Regularization** | Dropdown | Chooses the penalty type (`l2`, `l1`, `elasticnet`, `none`). |
| **C** | Number Input | Inverse of regularization strength; smaller values specify stronger regularization. |
| **Solver** | Dropdown | Choose optimization algorithm (`lbfgs`, `newton-cg`, `liblinear`, `sag`, `saga`). |
| **Max Iterations** | Number Input | Maximum number of iterations taken for the solvers to converge. |
| **Multi Class Strategy** | Dropdown | Implements standard multinomial or explicit One-Vs-Rest (`ovr`) strategy. |
| **l1 Ratio** | Number Input | Balancing parameter between L1 and L2 penalties (only active for `elasticnet`). |

---

## 🛡️ Built-in Parameter Guardrails

To prevent internal execution crashes, the application automatically handles scikit-learn version constraints and structural mismatches:
* **ElasticNet Guard**: Flags an error if `elasticnet` is selected with any solver other than `saga`.
* **L1 Guard**: Flags an error if `l1` is matched with incompatible solvers like `lbfgs`.
* **Scikit-Learn v1.7+ Compatibility**: Implicitly handles multiclass routing logic instead of relying on deprecated parameters.

---

## 📂 Project Structure

```text
├── Tool.py          # Main Streamlit application script
└── README.md        # Project setup and documentation guide
```
