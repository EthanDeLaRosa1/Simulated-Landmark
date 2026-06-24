"""
pose.py — compute yaw / pitch / roll from MediaPipe face landmarks
Uses a simplified PnP approach with 6 canonical face points.
"""

import numpy as np
import cv2


# Canonical 3D face model points (nose tip, chin, eye corners, mouth corners)
# in a neutral, forward-facing pose. Units are arbitrary.
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),      # nose tip
    (0.0,   -63.6, -12.5),      # chin
    (-43.3,  32.7, -26.0),      # left eye outer corner
    (43.3,   32.7, -26.0),      # right eye outer corner
    (-28.9, -28.9, -24.1),      # left mouth corner
    (28.9,  -28.9, -24.1),      # right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


def estimate_pose(
    landmarks,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float]:
    """
    Returns (yaw, pitch, roll) in degrees.
    landmarks: mediapipe NormalizedLandmarkList
    """
    if landmarks is None:
        return (0.0, 0.0, 0.0)

    # Build 2D image points from selected landmarks
    image_points = []
    lm = landmarks.landmark
    for idx in LANDMARK_INDICES:
        x = lm[idx].x * image_w
        y = lm[idx].y * image_h
        image_points.append((x, y))
    image_points = np.array(image_points, dtype=np.float64)

    # Camera intrinsics (estimated for a typical webcam)
    focal = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return (0.0, 0.0, 0.0)

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat([rotation_mat, np.zeros((3, 1))])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = float(euler[0].flat[0])
    yaw   = float(euler[1].flat[0])
    roll  = float(euler[2].flat[0])

    # clamp to readable range
    yaw   = max(-90.0, min(90.0,  yaw))
    pitch = max(-90.0, min(90.0,  pitch))
    roll  = max(-90.0, min(90.0,  roll))

    return (yaw, pitch, roll)
