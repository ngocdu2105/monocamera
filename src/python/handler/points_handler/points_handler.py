import numpy as np
import cv2


class PointProcessorHandler:
    """
    Compute the translation vector that maps the top-face center points to the bottom-face
    center points of an object using edge pixels extracted from images.

    This class overloads the `|` operator to compute a translation vector between two
    `PointProcessorHandler` objects. The operator syntax is defined as:

        >>> vector_top = PointProcessorHandler(edges_top)
        >>> vector_bottom = PointProcessorHandler(edges_bottom)
        >>> move_vector = vector_top | vector_bottom

    Args:
        edges (np.ndarray): Binary image (edge map) representing object edges. Non-zero
                            pixels are treated as 2D point coordinates.

    Raises:
        ValueError: If `edges` is not a numpy array or no non-zero points are found.
    """

    def __init__(self, edges: np.ndarray):
        if not isinstance(edges, np.ndarray):
            raise ValueError("Input must be a numpy.ndarray representing a cv::Mat")

        nonzero_pts = cv2.findNonZero(edges)
        if nonzero_pts is None:
            raise ValueError("Input edge map contains no edge points")

        # Convert to (N, 2) array of (x, y) coordinate pairs
        self.point_edges = np.array(np.where(edges > 0))[::-1].T

    # ----------------------------------------------------------------------
    # Operator overload: top_handler | bottom_handler
    # ----------------------------------------------------------------------
    def __or__(self, other):
        """
        Overload the `|` operator to compute a translation vector between two sets of points.
        The internal logic computes:
            - point intersections,
            - non-intersection points,
            - POI ↔ NIP vector relationships,
            - the optimal top-to-bottom translation vector.

        Args:
            other (PointProcessorHandler): The second point set to compare with.

        Returns:
            np.ndarray: A (2,) vector representing the translation between top and bottom.

        Raises:
            ValueError: If input is not the same class.
        """
        if not isinstance(other, PointProcessorHandler):
            raise ValueError("Operand must be a PointProcessorHandler instance")

        return self._compute_translation_vector(other)

    # ----------------------------------------------------------------------
    # Internal computation (refactored for readability)
    # ----------------------------------------------------------------------
    def _compute_translation_vector(self, other):
        # Pairwise distances (top → bottom)
        dist_top_bottom = np.linalg.norm(
            other.point_edges - self.point_edges[:, None], axis=-1
        )

        # Points considered "intersection" by threshold rule
        idx_intersection = np.array(
            list(set(np.argmax(dist_top_bottom < 4, axis=-1)))
        )[1:]

        idx_non_intersection = np.setxor1d(
            idx_intersection, np.arange(len(other.point_edges))
        )

        # Vector POI → NIP
        vec_poi_to_nip = (
            other.point_edges[idx_intersection]
            - other.point_edges[idx_non_intersection][:, None]
        )

        # Compute magnitude of each vector
        mag_poi_nip = np.linalg.norm(vec_poi_to_nip, axis=-1)

        # Select "max-min" index (as in original algorithm)
        idx_max_min_nip = np.argmax(np.sort(mag_poi_nip)[:, 1])
        idx_max_min_poi = np.argsort(mag_poi_nip)[:, 1][idx_max_min_nip]

        # Compute POI near base POI
        poi_base = other.point_edges[idx_intersection][idx_max_min_poi]

        value_poi_cluster = np.sum(
            abs(other.point_edges[idx_intersection] - poi_base), axis=-1
        )
        poi_cluster = other.point_edges[idx_intersection][value_poi_cluster < 100]

        # Vector cluster → NIP
        vec_cluster_to_nip = (
            other.point_edges[idx_non_intersection][idx_max_min_nip] - poi_cluster
        )
        moved_cluster = poi_cluster + vec_cluster_to_nip[:, None]

        # Choose best translation candidate
        mag_moved_vs_nip = np.linalg.norm(
            other.point_edges[idx_non_intersection] - moved_cluster[:, :, None],
            axis=-1,
        )

        best_idx = np.argmin(
            np.sum(np.min(mag_moved_vs_nip, axis=-1), axis=-1)
        )

        # Final translation vector
        final_vector = (
            -poi_cluster[best_idx]
            + other.point_edges[idx_non_intersection][idx_max_min_nip]
        )

        return final_vector