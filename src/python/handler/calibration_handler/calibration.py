import cv2
import numpy as np
import random
import os


class Calibrator:
    """
    Calibrator adjusts a specified pixel point based on its corresponding
    checkerboard coordinates.

    Args:
        path_calibration (str): Path to the image used for calibration.

    Attributes:
        CHECKERBOARD (tuple[int, int]): Checkerboard inner corner dimensions.
        imgpoints (np.ndarray): Detected subpixel corners from the checkerboard.
        pointOxy (np.ndarray): Reference pixel point on the checkerboard.
        dis_oxy (np.ndarray): Neighboring points used to calculate direction.
        oxy (list[np.ndarray]): Points used to draw results.
        appr_edge_length (float): Approximated size of one checkerboard square.
        __referenced_point (np.ndarray): Calibrated point in chessboard coordinates.
    """

    def __init__(self, path_calibration: str, checkerboard_dims: tuple[int, int] = (13, 9)):
        if not os.path.exists(path_calibration):
            raise FileNotFoundError(f"Calibration image not found: {path_calibration}")

        self.img = cv2.imread(path_calibration)
        self.CHECKERBOARD = checkerboard_dims
        self.__referenced_point = np.zeros(2)
        self.point3D = None

        self.__build_calibration()

    # -------------------- Internal methods --------------------
    def __build_calibration(self):
        """Detect and refine chessboard corners, calculate reference points."""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            self.CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not ret:
            raise RuntimeError(
                "Calibration failed: checkerboard corners not found. "
                "Check the image path or checkerboard dimensions."
            )

        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        self.imgpoints = refined_corners.reshape(-1, 2)

        # Reference point
        self.pointOxy = self.imgpoints[3]

        # Index of closest point to reference
        self.index_min = np.argmin(abs(self.imgpoints - self.pointOxy), axis=0)[0]

        # Neighbor points for direction calculation
        self.dis_oxy = np.array([
            self.imgpoints[self.index_min + 1],
            self.imgpoints[self.index_min + self.CHECKERBOARD[0] * 2],
        ])

        # Points used for visualization
        self.oxy = [
            self.pointOxy,
            self.imgpoints[self.index_min + 1],
            self.imgpoints[self.index_min + self.CHECKERBOARD[0] * 2],
        ]

        # Approximate edge length
        self.appr_edge_length = self.__estimate_square_size(refined_corners)

        print(f"Calibration loaded successfully. Approx. square size: {self.appr_edge_length:.2f}")

    def __estimate_square_size(self, corners: np.ndarray) -> float:
        """Estimate the average size of a checkerboard square."""
        idx = np.random.randint(len(corners), size=(len(corners), 1))
        distances = np.sqrt(np.sum((np.array(corners[idx]) - np.array(corners)) ** 2, axis=-1))
        return np.mean(np.sort(distances)[:, 1])

    def __coordinate_direction(self) -> np.ndarray:
        """Calculate direction along Oxy axis based on reference points."""
        angle_flags = self.__direction_angle(self.dis_oxy - self.pointOxy, self.point3D - self.pointOxy)
        return np.array([1 if flag else -1 for flag in angle_flags])

    def __direction_angle(self, vectors_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
        """Calculate angle between each vector in vectors_a and vector_b."""
        vectors_a = np.atleast_2d(vectors_a)
        vector_b = np.atleast_1d(vector_b)
        cos_angles = np.clip(np.sum(vectors_a * vector_b, axis=1) /
                             (np.linalg.norm(vectors_a, axis=1) * np.linalg.norm(vector_b)), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))
        return angles_deg < 90

    # -------------------- Public methods --------------------
    def reference_point_oxy(self, point3D: np.ndarray):
        """
        Convert a 3D point to chessboard coordinates relative to the reference point.

        Args:
            point3D (np.ndarray): 3D point to be calibrated.
        """
        distances = np.array([
            self.__distance_from_point_to_others(p2=self.pointOxy, p1=self.dis_oxy[0], p3=point3D) / self.appr_edge_length,
            self.__distance_from_point_to_others(p2=self.pointOxy, p1=self.dis_oxy[1], p3=point3D) / self.appr_edge_length
        ])

        # Swap coordinates if checkerboard is wider than tall
        if self.CHECKERBOARD[0] > self.CHECKERBOARD[1]:
            distances = distances[::-1]

        self.point3D = point3D
        self.__referenced_point = distances * self.__coordinate_direction()

    def draw_results(self, img: np.ndarray, radius: int = 5, color: tuple[int, int, int] = (0, 0, 255),
                     thickness: int = -1) -> np.ndarray:
        """
        Draw reference and calibration points on the image.

        Args:
            img (np.ndarray): Image to draw on.
            radius (int): Radius of the circles.
            color (tuple[int, int, int]): Color of points.
            thickness (int): Thickness of circle (-1 for filled).

        Returns:
            np.ndarray: Image with drawn points and text.
        """
        cv2.putText(
            img,
            f"P({self.__referenced_point[0]:0.2f},{self.__referenced_point[1]:0.2f})",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        for point in self.oxy:
            x, y = point
            cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        return img

    # -------------------- Utility methods --------------------
    @staticmethod
    def __distance_from_point_to_others(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Distance from point p3 to the line defined by points p1 and p2."""
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))