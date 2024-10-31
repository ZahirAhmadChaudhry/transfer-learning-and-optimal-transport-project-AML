from taipy.gui import Gui
import numpy as np

from data_loading import load_dataset
from preprocessing import preprocess_data
from domain_adaptation import subspace_alignment, optimal_transport
from models import get_classifier, train_classifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from visualization import plot_confusion_matrix, plot_data_projection
from utils import get_class_names

# Define global variables
source_domain = 'webcam'
source_domain_options = ['webcam', 'dslr', 'amazon', 'caltech']
target_domain = 'dslr'
target_domain_options = ['webcam', 'dslr', 'amazon', 'caltech']
feature_type = 'surf'
feature_type_options = ['surf', 'caffenet']
standardize = True
balance_classes = True
remove_outliers = True
da_method = 'Subspace Alignment'
da_method_options = ['None', 'Subspace Alignment', 'Optimal Transport']
d_value = 30
lambda_value = 1.0
classifier_type = '1-NN'
classifier_options = ['1-NN', 'SVM', 'Logistic Regression']
validation_technique = 'Cross-Validation'
validation_options = ['Cross-Validation', 'Hold-Out']
cv_folds = 5
accuracy = 0.0
confusion_fig = None
data_projection_fig = None

# Define individual functions for each step of the experiment
def load_and_preprocess_data(state):
    global X_S, y_S, X_T, y_T
    # Load datasets
    X_S, y_S = load_dataset(state.source_domain, state.feature_type)
    X_T, y_T = load_dataset(state.target_domain, state.feature_type)

    # Preprocess data
    X_S, y_S = preprocess_data(X_S, y_S, state.standardize, state.balance_classes, state.remove_outliers)
    X_T, y_T = preprocess_data(X_T, y_T, state.standardize, state.balance_classes, state.remove_outliers)

def run_domain_adaptation(state):
    global X_S_aligned, X_T_aligned, data_projection_fig
    # Perform domain adaptation
    if state.da_method == 'Subspace Alignment':
        X_S_aligned, X_T_pca = subspace_alignment(X_S, X_T, state.d_value)
        X_T_aligned = X_T_pca  # Projected target data
    elif state.da_method == 'Optimal Transport':
        X_S_aligned, X_T_aligned = optimal_transport(X_S, X_T, state.lambda_value)
    else:
        X_S_aligned, X_T_aligned = X_S, X_T

    # Dimensionality reduction for visualization
    pca_vis = PCA(n_components=2)
    X_S_vis = pca_vis.fit_transform(X_S_aligned)
    X_T_vis = pca_vis.transform(X_T_aligned)

    # Combine source and target data for visualization
    X_vis = np.vstack((X_S_vis, X_T_vis))
    domain_labels = np.array(['Source'] * len(y_S) + ['Target'] * len(y_T))

    # Plot data projection
    data_projection_fig = plot_data_projection(X_vis, domain_labels, title='Data Projection After Adaptation')

def train_and_evaluate_classifier(state):
    global accuracy, confusion_fig
    # Train classifier
    model = get_classifier(state.classifier_type)
    model = train_classifier(model, X_S_aligned, y_S)

    # Predict on target data
    y_pred = model.predict(X_T_aligned)
    accuracy = accuracy_score(y_T, y_pred) * 100

    # Plot confusion matrix
    class_names = get_class_names(np.concatenate((y_S, y_T)))
    confusion_fig = plot_confusion_matrix(y_T, y_pred, class_names)

# Define the page content
page = '''
# Domain Adaptation Application

## Dataset Selection
<|{source_domain}|selector|lov={source_domain_options}|label=Select Source Domain|>
<|{target_domain}|selector|lov={target_domain_options}|label=Select Target Domain|>
<|{feature_type}|selector|lov={feature_type_options}|label=Select Feature Type|>

## Preprocessing Options
<|{standardize}|toggle|label=Standardize Data|>
<|{balance_classes}|toggle|label=Handle Class Imbalance (SMOTE)|>
<|{remove_outliers}|toggle|label=Remove Outliers (Isolation Forest)|>

<|Load and Preprocess Data|button|on_action=load_and_preprocess_data|>

## Domain Adaptation Method
<|{da_method}|selector|lov={da_method_options}|label=Select Domain Adaptation Method|>

### Subspace Alignment Parameters
<|{d_value}|slider|min=1|max=100|step=1|label=Number of Principal Components (d)|render={da_method=='Subspace Alignment'}|>

### Optimal Transport Parameters
<|{lambda_value}|slider|min=0.01|max=10.0|step=0.01|label=Entropic Regularization Parameter (λ)|render={da_method=='Optimal Transport'}|>

<|Run Domain Adaptation|button|on_action=run_domain_adaptation|>

## Classifier and Validation
<|{classifier_type}|selector|lov={classifier_options}|label=Select Classifier|>
<|{validation_technique}|selector|lov={validation_options}|label=Select Validation Technique|>

<|{cv_folds}|slider|min=2|max=10|step=1|label=Number of CV Folds|render={validation_technique=='Cross-Validation'}|>

<|Train and Evaluate Classifier|button|on_action=train_and_evaluate_classifier|>

## Results
**Accuracy:** {accuracy:.2f}%

### Confusion Matrix
<|{confusion_fig}|image|>

### Data Projection
<|{data_projection_fig}|image|>
'''

# Initialize the GUI
gui = Gui(page=page)

# Run the GUI
if __name__ == '__main__':
    gui.run()