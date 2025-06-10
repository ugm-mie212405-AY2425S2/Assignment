import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet

class FaceNetClassifier:
    def __init__(self, model_path='facenet_svm_classifier.pkl'):
        self.embedder = FaceNet()
        self.classifier, self.label_encoder = joblib.load(model_path)

    def preprocess(self, image):
        image = cv2.resize(image, (160, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.asarray(image)

    def predict(self, image):
        preprocessed = self.preprocess(image)
        embedding = self.embedder.embeddings([preprocessed])[0]
        
        proba = self.classifier.predict_proba([embedding])[0]
        best_idx = np.argmax(proba)
        label = self.label_encoder.inverse_transform([best_idx])[0]
        confidence = proba[best_idx]
        
        return label, confidence
