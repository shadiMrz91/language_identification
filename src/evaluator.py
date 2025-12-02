from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import os

class Evaluator:
    def __init__(self, y_true, y_pred, labels):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels

    def metrics(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        f1_macro = f1_score(self.y_true, self.y_pred, average='macro')
        f1_weighted = f1_score(self.y_true, self.y_pred, average='weighted')
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

    def report(self):
        return classification_report(self.y_true, self.y_pred, target_names=self.labels)

    def save_report(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.report())

    def save_misclassified(self, texts, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame({"text": texts, "true": self.y_true, "pred": self.y_pred})
        df_mis = df[df['true'] != df['pred']]
        df_mis.to_csv(file_path, index=False, encoding="utf-8")
        return len(df_mis)
    
    def get_misclassified_samples(self, texts, original_labels, label_encoder):        
        misclassified_indices = [i for i in range(len(self.y_true)) if self.y_true[i] != self.y_pred[i]]
        
        misclassified_data = []
        for idx in misclassified_indices:
            misclassified_data.append({
                'text': texts[idx],
                'true_label': original_labels[idx],
                'predicted_label': label_encoder.inverse_transform([self.y_pred[idx]])[0],
                'true_label_encoded': self.y_true[idx],
                'predicted_label_encoded': self.y_pred[idx]
            })
        
        return pd.DataFrame(misclassified_data)