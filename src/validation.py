# validation.py

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cross_validate_model(X, y, classifier_type, cv_folds):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = get_classifier(classifier_type)
        model = train_classifier(model, X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

    avg_acc = np.mean(accuracies)
    return avg_acc
