import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # [web:76]

EMB_FILE = "embeddings.npy"

def load_embeddings():
    X = np.load(EMB_FILE)
    print(f"[INFO] Loaded embeddings: {X.shape}")
    return X

def evaluate_kmeans(X, k_min=2, k_max=10):
    results = {}

    for k in range(k_min, k_max + 1):
        print(f"[INFO] Running KMeans for K={k}...")
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        print(f"  Silhouette(K={k}) = {sil:.4f}")

        results[k] = sil

    return results

def main():
    X = load_embeddings()
    results = evaluate_kmeans(X, k_min=2, k_max=10)

    ks = np.array(sorted(results.keys()))
    silhouettes = np.array([results[k] for k in ks])

    np.save("k_values.npy", ks)
    np.save("silhouette_values.npy", silhouettes)

    print("[INFO] Saved K values to k_values.npy")
    print("[INFO] Saved silhouette scores to silhouette_values.npy")

    best_k = ks[np.argmax(silhouettes)]
    best_sil = silhouettes.max()
    print(f"[INFO] Best K by static Silhouette: {best_k} (score={best_sil:.4f})")

if __name__ == "__main__":
    main()
