import logging

import seaborn as sns
from sklearn.decomposition import PCA

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from project_panaroma import Stitcher

logging.basicConfig(level=logging.INFO)


class HierarchicalClustering:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix.copy()
        self.num_images = distance_matrix.shape[0]
        self.clusters = [[i] for i in range(self.num_images)]
        self.distances = self._convert_similarity_to_distance(distance_matrix)

    def _convert_similarity_to_distance(self, similarity_matrix):
        # Convert similarity scores to distances
        # Adding epsilon to avoid division by zero
        epsilon = 1e-5
        distance_matrix = 1 / (similarity_matrix + epsilon)
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix

    def fit(self, num_clusters=1):
        while len(self.clusters) > num_clusters:
            # Find the two clusters with the smallest distance
            min_distance = np.inf
            x, y = -1, -1
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    dist = self._cluster_distance(self.clusters[i], self.clusters[j])
                    if dist < min_distance:
                        min_distance = dist
                        x, y = i, j
            if x == -1 or y == -1:
                break  # No more clusters to merge
            # Merge clusters x and y
            self.clusters[x].extend(self.clusters[y])
            del self.clusters[y]
            logging.info(
                f"Merged clusters {x} and {y}, new cluster: {self.clusters[x]}"
            )
        return self.clusters

    def _cluster_distance(self, cluster1, cluster2):
        # Compute the average linkage distance between two clusters
        distances = [self.distances[i][j] for i in cluster1 for j in cluster2]
        return np.mean(distances)
    
    def plot_pca_feature_space(self, distance_matrix):
        # Apply PCA to reduce the distance matrix to 2D
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(distance_matrix)

        # Plot the 2D feature space
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100)

        # Label each point with its corresponding image index
        for i in range(pca_result.shape[0]):
            plt.text(pca_result[i, 0], pca_result[i, 1], f"Image {i+1}",
                     fontsize=12, ha="center")

        plt.title("PCA of Feature Space (2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()


class StitcherWithRecommender(Stitcher):
    def compute_pairwise_similarities(self):
        num_images = len(self.input_images)
        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            kp1, des1 = self.feature_points_and_descriptors[i]
            for j in range(i + 1, num_images):
                kp2, des2 = self.feature_points_and_descriptors[j]
                good_matches = self.match_features_call(des1, des2)
                similarity_score = len(good_matches)
                similarity_matrix[i, j] = similarity_score
                similarity_matrix[j, i] = similarity_score  # Symmetric matrix
                print(
                    f"Similarity between image {i+1} and image {j+1}: {similarity_score} good matches"
                )

        return similarity_matrix

    def plot_distance_matrix_heatmap(self, distance_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                    xticklabels=[f"Image {i+1}" for i in range(distance_matrix.shape[0])],
                    yticklabels=[f"Image {i+1}" for i in range(distance_matrix.shape[0])])
        plt.title("Distance Matrix Heatmap")
        plt.xlabel("Images")
        plt.ylabel("Images")
        plt.show()
    
    def generate_dendrogram(self, similarity_matrix):
        # Convert the similarity matrix to a distance matrix
        epsilon = 1e-5  # Small value to prevent division by zero
        distance_matrix = 1 / (similarity_matrix + epsilon)
        np.fill_diagonal(distance_matrix, 0)  # Ensure the diagonal is zero

        # Convert the distance matrix to condensed form
        condensed_distance = squareform(distance_matrix)

        # Generate the linkage matrix using average linkage
        linked = linkage(condensed_distance, method="average")

        fig, ax = plt.subplots(figsize=(10, 7))
        dendrogram_data = dendrogram(
            linked,
            labels=[f"Image {i+1}" for i in range(similarity_matrix.shape[0])],
            ax=ax,
        )
        plt.title("Dendrogram of Image Similarities")
        plt.xlabel("Images")
        plt.ylabel("Distance")

        # Load images and display them below the dendrogram
        num_labels = len(dendrogram_data["ivl"])
        image_height = 0.1  # Adjust as needed

        # Get the x positions of the leaves
        xlbls = ax.get_xticks()
        xlbls = ax.get_xticks()

        for i, label in enumerate(dendrogram_data["ivl"]):
            image_index = int(label.split()[-1]) - 1
            img = cv2.cvtColor(self.input_images[image_index], cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (50, 50))

            # Get the x position of the label
            x = xlbls[i]
            y = 0  # y-position can be zero as we're only interested in x

            # Transform x position from data to figure coordinates
            trans = ax.transData.transform((x, y))
            inv = fig.transFigure.inverted()
            x_fig, y_fig = inv.transform(trans)

            # Width of each image in figure coordinates
            width = (
                1 / num_labels * 0.8
            )  # Adjust 0.8 to increase spacing between images
            x_fig_centered = x_fig - width / 2

            # Add new axes at the calculated position
            img_ax = fig.add_axes(
                [x_fig_centered, 0.01, width, image_height], anchor="S", zorder=1
            )
            img_ax.imshow(img)
            img_ax.axis("off")

        plt.subplots_adjust(bottom=0.3)  # Adjust to make space for images
        plt.show()

    def recommend_images(self, num_clusters=1):
        similarity_matrix = self.compute_pairwise_similarities()
        self.generate_dendrogram(similarity_matrix)  # dendrogram before clustering
        hc = HierarchicalClustering(similarity_matrix)
        clusters = hc.fit(num_clusters=num_clusters)
        recommendations = []
        for idx, cluster in enumerate(clusters):
            print(f"Cluster {idx + 1}: {[i + 1 for i in cluster]}")
            recommendations.append([self.input_images[i] for i in cluster])
        return recommendations

    def stitch_recommended_images(self, recommended_image_groups):
        for idx, image_group in enumerate(recommended_image_groups):
            self.input_images = image_group
            self.feature_points_and_descriptors = []
            self.detect_keypoints_and_descriptors()
            self.stitch3_with_post()
            print(f"Stitched panorama for cluster {idx + 1} saved.")


# Assuming `stitcher` is an instance of the Stitcher class
stitcher = StitcherWithRecommender()
stitcher.input_dir = "data/gallery_of_3s"
stitcher.output_dir = "data/gallery_of_3s_outputs"
stitcher.plot = False

stitcher.read_input_dir()
stitcher.detect_keypoints_and_descriptors()


# Compute the similarity matrix for the images
similarity_matrix = stitcher.compute_pairwise_similarities()

# Create an instance of HierarchicalClustering to convert similarity to distance
hc = HierarchicalClustering(similarity_matrix)
distance_matrix = hc._convert_similarity_to_distance(similarity_matrix)

# Plot the heatmap of the initial distance matrix
stitcher.plot_distance_matrix_heatmap(distance_matrix)

recommended_image_groups = stitcher.recommend_images(
    num_clusters=len(stitcher.input_images) // 3
)
stitcher.stitch_recommended_images(recommended_image_groups)
similarity_matrix = stitcher.compute_pairwise_similarities()

hc = HierarchicalClustering(similarity_matrix)

distance_matrix = hc._convert_similarity_to_distance(similarity_matrix)

stitcher.plot_distance_matrix_heatmap(distance_matrix)

