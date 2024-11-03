import numpy as np

from Stitcher import Stitcher


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
            print(f"Merged clusters {x} and {y}, new cluster: {self.clusters[x]}")
        return self.clusters

    def _cluster_distance(self, cluster1, cluster2):
        # Compute the average linkage distance between two clusters
        distances = [self.distances[i][j] for i in cluster1 for j in cluster2]
        return np.mean(distances)


class StitcherWithRecommender(Stitcher):
    def compute_pairwise_similarities(self):
        num_images = len(self.input_images)
        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            kp1, des1 = self.feature_points_and_descriptors[i]
            for j in range(i + 1, num_images):
                kp2, des2 = self.feature_points_and_descriptors[j]
                good_matches = self.match_features(des1, des2)
                similarity_score = len(good_matches)
                similarity_matrix[i, j] = similarity_score
                similarity_matrix[j, i] = similarity_score  # Symmetric matrix
                print(
                    f"Similarity between image {i+1} and image {j+1}: {similarity_score} good matches"
                )

        return similarity_matrix

    def recommend_images(self, num_clusters=1):
        similarity_matrix = self.compute_pairwise_similarities()
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
            self.align_and_stitch_images()
            print(f"Stitched panorama for cluster {idx + 1} saved.")


# Assuming `stitcher` is an instance of the Stitcher class
stitcher = StitcherWithRecommender(
    input_dir="data/hostel_room_sequence",
    output_dir="data/outputs",
    feature_detector="SIFT",
    matcher_type="BF",
    plot=True,
)
stitcher.read_input_dir()
stitcher.detect_keypoints_and_descriptors()
recommended_image_groups = stitcher.recommend_images(num_clusters=2)
stitcher.stitch_recommended_images(recommended_image_groups)
