import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Job_Placement_Prediction")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

eval_data = pd.DataFrame(X_test, columns=["f1", "f2", "f3", "f4"])
eval_data["label"] = y_test

# Define multiple models to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "SVM": SVC(kernel="rbf", probability=True),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        mlflow.log_param("model_type", name)
        mlflow.log_params(model.get_params())

        model_info = mlflow.sklearn.log_model(model, "model")

        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_data,
            targets="label",
            model_type="classifier",
        )

        print(f"{name} logged successfully")
