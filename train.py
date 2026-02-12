import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Job_Placement_Prediction")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    mlflow.log_param("max_iter", 100)

    model_info = mlflow.sklearn.log_model(model, "model")

    # mlflow.evaluate auto-logs accuracy, precision, recall, f1, log_loss, etc.
    eval_data = pd.DataFrame(X_test, columns=["f1", "f2", "f3", "f4"])
    eval_data["label"] = y_test

    mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets="label",
        model_type="classifier",
    )
