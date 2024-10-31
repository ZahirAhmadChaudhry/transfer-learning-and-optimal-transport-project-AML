# models.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_classifier(classifier_type):
    if classifier_type == '1-NN':
        model = KNeighborsClassifier(n_neighbors=1)
    elif classifier_type == 'SVM':
        model = SVC(kernel='linear', random_state=42)
    elif classifier_type == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError("Invalid classifier type.")
    return model

def train_classifier(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
