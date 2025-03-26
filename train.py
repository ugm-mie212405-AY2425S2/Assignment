import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from utils import extract_features

# Load dataset
X, y = [], []
genres = os.listdir("genres")  # Pastikan folder "genres" ada

for genre in genres:
    genre_path = os.path.join("genres", genre)
    if os.path.isdir(genre_path):
        print(f"Processing genre: {genre}")
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(genre)

X = np.array(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 **Akurasi Model setelah Hypertuning:** {accuracy * 100:.2f}%\n")

# Laporan klasifikasi
print("📊 **Laporan Klasifikasi:**")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Simpan model, label encoder, dan scaler
# joblib.dump(best_model, "random_forest_model.pkl")
# joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model berhasil dilatih dan disimpan!")
