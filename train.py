import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prometheus_client import start_http_server, Gauge
import time

# Start Prometheus metrics server on port 8000
start_http_server(8000)
print("Prometheus metrics server started on http://localhost:8000/metrics")

# Create Prometheus metrics
accuracy_metric = Gauge('model_accuracy', 'Model accuracy score', ['model_name'])
precision_metric = Gauge('model_precision', 'Model precision score', ['model_name'])
recall_metric = Gauge('model_recall', 'Model recall score', ['model_name'])
f1_metric = Gauge('model_f1_score', 'Model F1 score', ['model_name'])
training_time = Gauge('model_training_time_seconds', 'Model training time', ['model_name'])

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "SVM": SVC(kernel="rbf", probability=True),
}

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Export to Prometheus
    accuracy_metric.labels(model_name=name).set(accuracy)
    precision_metric.labels(model_name=name).set(precision)
    recall_metric.labels(model_name=name).set(recall)
    f1_metric.labels(model_name=name).set(f1)
    training_time.labels(model_name=name).set(train_duration)
    
    print(f"{name} metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training Time: {train_duration:.4f}s")

print("\n✅ All models trained and metrics exposed!")
print("📊 View metrics at: http://localhost:8000/metrics")
print("🔄 Keeping server running... Press Ctrl+C to stop")

# Keep the server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nServer stopped")