import os
import time
import pandas as pd
from embedding_extractor import EmbeddingExtractor
from evaluator import Evaluator
from xlmroberta_predictor import XLMRobertaPredictor

class TestPipeline:
    def __init__(self):
        self.times = {}             
        self.wrong_predictions = {}   

    def run_test(self, classifiers, test_texts, test_labels, label_encoder, timestamp):
        results = []
        y_test = label_encoder.transform(test_labels)

        for name, clf in classifiers.items():
            results.append(self._test_model(name, clf, test_texts, test_labels, label_encoder, y_test, timestamp))


        results.append(self._test_xlmr(test_texts, test_labels, label_encoder, y_test, timestamp))

        self._save_all(timestamp, results)
 
        final_df = self._combine_results(results)
        final_df.to_csv(f"results/final_predictions_{timestamp}.csv", index=False, encoding="utf-8")
        print(f"\n All predictions saved to results/final_predictions_{timestamp}.csv")
        return final_df


    def _test_model(self, name, clf, texts, labels, le, y_true, timestamp):
        start = time.time()
        extractor = EmbeddingExtractor(model_name=name)
        X_test = extractor.compute_embeddings(texts)
        y_pred, _ = clf.predict(X_test)
        self.times[name] = time.time() - start

        ev = Evaluator(y_true, y_pred, labels=le.classes_)
        self.wrong_predictions[name] = ev.get_misclassified_samples(texts, labels, le)

        metrics = ev.metrics()
        print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['f1_macro']:.4f}, "
              f"Weighted-F1={metrics['f1_weighted']:.4f}, Time={self.times[name]:.2f}s")
     
        ev.save_report(f"results/{name}_report_{timestamp}.txt")
        self.wrong_predictions[name].to_excel(f"results/{name}_misclassified_{timestamp}.xlsx", index=False)

        return {"text": texts, "true": labels, f"{name}_pred": le.inverse_transform(y_pred)}
    

    def _test_xlmr(self, texts, labels, le, y_true, timestamp):
        start = time.time()
        predictor = XLMRobertaPredictor(label_encoder=le)
        y_pred = predictor.predict(texts)
        self.times['XLM-R'] = time.time() - start

        ev = Evaluator(y_true, y_pred, labels=le.classes_)
        self.wrong_predictions['XLM-R'] = ev.get_misclassified_samples(texts, labels, le)

        metrics = ev.metrics()
        print(f"XLM-R: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['f1_macro']:.4f}, "
              f"Weighted-F1={metrics['f1_weighted']:.4f}, Time={self.times['XLM-R']:.2f}s")

        ev.save_report(f"results/XLM-R_report_{timestamp}.txt")
        self.wrong_predictions['XLM-R'].to_excel(f"results/XLM-R_misclassified_{timestamp}.xlsx", index=False)

        return {"text": texts, "true": labels, "xlmr_pred": le.inverse_transform(y_pred)}


    def _save_all(self, timestamp, results):
        pd.DataFrame(list(self.times.items()), columns=['model', 'time']).to_csv(
            f"results/times_{timestamp}.csv", index=False
        )
        print(f" Execution times saved to results/times_{timestamp}.csv")



    def _combine_results(self, results):
        df = pd.DataFrame(results[0])
        for r in results[1:]:
            for col, values in r.items():
                if col not in df.columns:
                    df[col] = values
        return df