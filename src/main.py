import os
from datetime import datetime
from data_handler import DataHandler
from embedding_extractor import EmbeddingExtractor
from classifier import ClassifierModel
from sklearn.preprocessing import LabelEncoder
from test_pipeline import TestPipeline
from visualizer import Visualizer 

def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dh = DataHandler()
    data = dh.prepare_data()
    train_texts, train_labels = data['train']
    dev_texts, dev_labels = data['dev']
    test_texts, test_labels = data['test']

    le = LabelEncoder()
    le.fit(train_labels)

    classifiers = {}
    for model_name in ["labse", "distilmbert"]:
        print(f"\n Starting pipeline for embedding model: {model_name}")

        extractor = EmbeddingExtractor(model_name=model_name)
        X_train = extractor.load_or_compute_embeddings(train_texts, "train")
        X_dev = extractor.load_or_compute_embeddings(dev_texts, "dev")

        y_train = le.transform(train_labels)
        y_dev = le.transform(dev_labels)

        vis = Visualizer(X_train, train_labels)
        vis.plot_tsne(f"results/{model_name}_train_tsne.png", title=f"{model_name} Train Embeddings")

        clf = ClassifierModel()
        model_path = f"models/{model_name}_classifier.pkl"

        if os.path.exists(model_path):
            print(f" Loading pre-trained classifier for {model_name} ...")
            clf.load(model_name)
        else:
            print(f" Training logistic regression classifier for {model_name} ...")
            clf.train(X_train, y_train, X_dev, y_dev)
            clf.save(model_name)

        classifiers[model_name] = clf

    test_pipeline = TestPipeline()
    final_results = test_pipeline.run_test(
        classifiers,
        test_texts,
        test_labels,
        le,
        timestamp
    )
    print("\n Test pipeline completed successfully!")
    print("\n Main pipeline completed successfully!")

if __name__ == "__main__":
    main()