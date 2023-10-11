from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load data
X = np.load("release_gene_analysis_data/data/p1/X.npy")
y = np.load("release_gene_analysis_data/data/p1/y.npy")

# Apply a log-transformation to the data
X_log = np.log2(X + 1)

# 1. Data preparation using PCA for 100 principal components
pca_100 = PCA(n_components=100)
X_pca_100 = pca_100.fit_transform(X_log)

# 2. Experiment with T-SNE learning rate
# --------------------------------------
for lr in [10, 20, 50, 100, 200, 500, 1000]:
    tsne = TSNE(n_components=2, learning_rate=lr)
    X_tsne = tsne.fit_transform(X_pca_100)
    
    plt.figure()
    plt.title(f'T-SNE with learning rate = {lr}')
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    # Save the plot in the 'outcomes' folder
    plt.savefig(f'outcomes/tsne_{lr}_learning_rate.png')
    # plt.show()
    plt.close()

# 3. Experiment with T-SNE number of iterations/convergence tolerance
# ------------------------------------------------------------------
for n_iter in [250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
    tsne = TSNE(n_components=2, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X_pca_100)
    
    plt.figure()
    plt.title(f'T-SNE with number of iterations = {n_iter}')
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    # Save the plot in the 'outcomes' folder
    plt.savefig(f'outcomes/tsne_{n_iter}_n_iter.png')
    # plt.show()
    plt.close()


# 4. Examine the effect of the number of principal components chosen on clustering
# -------------------------------------------------------------------------------
for n_components in [10, 20, 50, 100, 200, 300, 400, 500]:
     # Perform a new PCA for each number of principal components
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_log)
    
    # Clustering
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X_reduced)
    
    # Visualization (using only the first two principal components)
    plt.figure()
    plt.title(f'K-means clusters with {n_components} principal components')
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_)
    
    # Save the plot
    plt.savefig(f'outcomes/kmeans_{n_components}_PCs.png')
    plt.close()
    print(f'Inertia for {n_components} principal components: {kmeans.inertia_}')
