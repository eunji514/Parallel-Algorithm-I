import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler


class AffinityPropagationClusterer:

    # Initialize the Affinity Propagation clustering algorithm.
    def __init__(self, damping=0.5, max_iterations=200, min_iterations=15):
        self.damping = damping
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
    
    # Calculate the similarity matrix for the input data.
    def _compute_similarity_matrix(self, data):
        n_points = data.shape[0]
        similarity = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                similarity[i, j] = -np.sum((data[i] - data[j]) ** 2)
        
        np.fill_diagonal(similarity, np.mean(similarity))
        
        return similarity
    
    # Fit the Affinity Propagation clustering algorithm to the input data.
    def fit(self, data):
        n_points = data.shape[0]
        similarity_matrix = self._compute_similarity_matrix(data)
        
        # Initialize responsibility and availability matrices
        responsibility_matrix = np.zeros((n_points, n_points))
        availability_matrix = np.zeros((n_points, n_points))
        
        for iteration in range(self.max_iterations):

            # Store previous responsibility and availability matrices
            prev_responsibility = responsibility_matrix.copy()
            prev_availability = availability_matrix.copy()
            
            # Update responsibility matrix
            for i in range(n_points):
                for k in range(n_points):
                    max_other = np.max(np.concatenate([
                        similarity_matrix[i, :k],
                        similarity_matrix[i, k+1:]
                    ]) + np.concatenate([
                        prev_availability[i, :k],
                        prev_availability[i, k+1:]
                    ]))
                    responsibility_matrix[i, k] = (1 - self.damping) * (similarity_matrix[i, k] - max_other) + self.damping * prev_responsibility[i, k]
            
            # Update availability matrix
            for i in range(n_points):
                for k in range(n_points):
                    if i != k:
                        r_sum = np.sum(np.maximum(0, responsibility_matrix[:, k]))
                        availability_matrix[i, k] = (1 - self.damping) * min(0, responsibility_matrix[k, k] + r_sum - max(0, responsibility_matrix[i, k])) + self.damping * prev_availability[i, k]
            
            # Update diagonal availability
            for k in range(n_points):
                availability_matrix[k, k] = (1 - self.damping) * np.sum(np.maximum(0, responsibility_matrix[:, k])) + self.damping * prev_availability[k, k]
            
            # Check for convergence
            if iteration >= self.min_iterations and np.allclose(responsibility_matrix, prev_responsibility) and np.allclose(availability_matrix, prev_availability):
                break
        
        # Determine cluster representatives and assignments
        criterion_matrix = responsibility_matrix + availability_matrix
        cluster_labels = np.argmax(criterion_matrix, axis=1)
        cluster_representatives = np.unique(cluster_labels)
        
        # Create cluster assignments dictionary
        cluster_assignments = {}
        for rep in cluster_representatives:
            cluster_assignments[rep] = np.where(cluster_labels == rep)[0]
        
        return cluster_labels, cluster_representatives, cluster_assignments

# Load and preprocess MNIST data.
def load_mnist_data(file_path, num_samples=500, random_state=42):
    df = pd.read_csv(file_path)
    df_sample = df.sample(n=num_samples, random_state=random_state)
    
    data = df_sample.iloc[:, 1:].values
    labels = df_sample.iloc[:, 0].values
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, labels

# Visualize the clustering results using UMAP.
def visualize_clusters(data, cluster_labels, cluster_representatives, file_name='mnist_clusters', title='Clustering Results'):
    
    # Reduce the data to 2D using UMAP
    reducer = umap.UMAP(n_neighbors=8, min_dist=0.1, n_components=2)
    reduced_data = reducer.fit_transform(data)
    
    plt.figure(figsize=(15, 10))
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        cluster_points = reduced_data[cluster_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.7)
        
        # Highlight the cluster representatives
        rep_index = np.where(cluster_labels == label)[0][0]
        rep_point = reduced_data[rep_index]
        plt.scatter(rep_point[0], rep_point[1], color='black', marker='x', s=200, linewidths=3)
    
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    
    # For save the figure
    if file_name: 
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    
    # Load MNIST data
    file_path = 'archive/mnist_test.csv'
    data, labels = load_mnist_data(file_path)
    
    # Apply Affinity Propagation clustering
    clusterer = AffinityPropagationClusterer(damping=0.5, max_iterations=200, min_iterations=15)
    cluster_labels, cluster_representatives, cluster_assignments = clusterer.fit(data)
    
    # Print clustering results
    print(f"Number of discovered clusters: {len(cluster_representatives)}")
    for rep_idx, cluster_points in cluster_assignments.items():
        print(f"Cluster representative {rep_idx}: {len(cluster_points)} data points")
        print("Included labels:", np.unique(labels[cluster_points]))
    
    # Visualize the clustering results
    visualize_clusters(data, cluster_labels, cluster_representatives)

if __name__ == "__main__":
    main()