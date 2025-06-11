import pickle
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load encoded data
with open("face_data.pkl", 'rb') as f:
    data = pickle.load(f)
    
X = data["encodings"]
y = data["labels"]

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Predict and evaluate
preds = clf.predict(X_test)
accuracy = accuracy_score(y_test, preds)
conf_matrix = confusion_matrix(y_test, preds)
class_report = classification_report(y_test, preds, output_dict=True)

print("[INFO] Accuracy:", accuracy)
print("[INFO] Confusion Matrix:\n", conf_matrix)
print("[INFO] Classification Report:\n", classification_report(y_test, preds))

# Save model
with open("svm_classifier.pkl", 'wb') as f:
    pickle.dump(clf, f)
    
# Save evaluation report to CSV
with open("evaluation_metrics.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Label", "Precision", "Recall", "F1-score", "Support"])
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            writer.writerow([
                label,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1-score", 0),
                metrics.get("support", 0)
            ])
    writer.writerow(["Accuracy", accuracy, "", "", ""])

print("[DONE] Model and evaluation saved.")