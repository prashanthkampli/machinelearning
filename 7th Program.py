import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

np.random.seed(0)
num_samples, num_features = 100, 2
data = np.random.rand(num_samples, num_features)
pd.DataFrame(data, columns=['X', 'Y']).to_csv('test_data.csv', index=False)
print("CSV file 'test_data.csv' has been generated successfully.")

data = pd.read_csv('test_data.csv').values
k = 3

kmeans = KMeans(n_clusters=k).fit(data)
kmeans_labels, kmeans_centers = kmeans.labels_, kmeans.cluster_centers_

em = GaussianMixture(n_components=k).fit(data)
em_labels, em_centers = em.predict(data), em.means_

print("K-means labels:", kmeans_labels)
print("EM labels:", em_labels)

plt.figure(figsize=(12, 5))
for i, (labels, centers, title) in enumerate([
    (kmeans_labels, kmeans_centers, 'K-means Clustering'),
    (em_labels, em_centers, 'EM Clustering')], 1):
    plt.subplot(1, 2, i)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=300, c='r')
    plt.title(title)
plt.show()
