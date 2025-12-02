from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
import os
import joblib

class ClassifierModel:
    def __init__(self, c_values=[0.01,0.1,1,10], model_dir="models"):
        self.c_values = c_values
        self.model = None
        self.best_c = None
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X_train, y_train, X_dev=None, y_dev=None):
        best_acc = 0
        best_model = None
        for c in self.c_values:
            model = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs', C=c)
            model.fit(X_train, y_train)
            if X_dev is not None and y_dev is not None:
                preds = model.predict(X_dev)
                acc = accuracy_score(y_dev, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    self.best_c = c
            else:
                best_model = model
                self.best_c = c
                break
        self.model = best_model

    def predict(self, X):
        start = time.time()
        y_pred = self.model.predict(X)
        duration = time.time() - start
        return y_pred, duration
    
    def save(self, model_name):
        path = os.path.join(self.model_dir, f"{model_name}_classifier.pkl")
        joblib.dump({"model": self.model, "best_c": self.best_c}, path)
        print(f" Saved model: {path}")

    def load(self, model_name):
        path = os.path.join(self.model_dir, f"{model_name}_classifier.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.best_c = data["best_c"]
        print(f" Loaded model: {path}")