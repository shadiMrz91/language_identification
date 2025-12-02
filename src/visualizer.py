import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.cm as cm


class Visualizer:
    def __init__(self, embeddings, labels):
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)

    def plot_tsne(self, save_path, title="t-SNE projection", max_points=3000):
        
        if len(self.embeddings) > max_points:
            idx = np.random.choice(len(self.embeddings), size=max_points, replace=False)
            emb = self.embeddings[idx]
            lbls = self.labels[idx]
            print(f" Too many samples, only {max_points} points will be used for t-SNE visualization.")
        else:
            emb = self.embeddings
            lbls = self.labels

        pca_dim = min(50, emb.shape[1])
        pca_emb = PCA(n_components=pca_dim).fit_transform(emb)

        reducer = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42)
        X_2d = reducer.fit_transform(pca_emb)

        plt.figure(figsize=(12, 8))
        unique_labels = list(set(lbls))
        colors = cm.get_cmap('tab20', len(unique_labels))

        for idx, label in enumerate(unique_labels):
            mask = lbls == label
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                label=label,
                color=colors(idx),
                s=10,
                alpha=0.7
            )

        plt.title(title)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Visualization saved: {save_path}")