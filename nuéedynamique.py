
import numpy as np
import matplotlib.pyplot as plt

class NueesDynamiques:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def distance_euclidienne(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. Initialisation aléatoire des centroïdes parmi les données
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Affectation des points au centroïde le plus proche
            clusters = [[] for _ in range(self.k)]
            labels = []
            
            for x in X:
                distances = [self.distance_euclidienne(x, c) for c in self.centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(x)
                labels.append(cluster_idx)
                
            self.labels = np.array(labels)
            
            # 3. Mise à jour des centroïdes
            anciens_centroids = np.copy(self.centroids)
            for k_idx in range(self.k):
                if clusters[k_idx]: # Éviter les clusters vides
                    self.centroids[k_idx] = np.mean(clusters[k_idx], axis=0)
                    
            # 4. Condition d'arrêt (convergence)
            if np.all(anciens_centroids == self.centroids):
                break

# --- Exemple d'utilisation avec des données factices ---
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    # Génération de données de test
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    
    # Application de l'algorithme
    kmeans = NueesDynamiques(k=3)
    kmeans.fit(X)
    
    print("Centroïdes finaux trouvés :")
    print(kmeans.centroids)