from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def run_sklearn_models(X_train, X_test, y_train, y_test):

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC()
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print(f"{name} Accuracy: {acc:.4f}")

        results[name] = acc

    return results