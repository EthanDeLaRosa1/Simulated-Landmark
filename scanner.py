"""
scanner.py — webcam capture loop + MediaPipe face mesh + rendering
"""

import cv2
import mediapipe as mp
import numpy as np
import time

from src.renderer import Renderer
from src.pose import estimate_pose


# ── Z depth range for normalization ──────────────────────────────────────────
# MediaPipe z is relative; typical range across all 468 landmarks is roughly
# -0.12 (close, nose tip) to +0.12 (far, ears/back). We clamp and normalize.
Z_NEAR = -0.15
Z_FAR  =  0.15


class FaceScanner:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ):
        self.cam_idx = camera_index
        self.width   = width
        self.height  = height

        # MediaPipe setup
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # enables iris landmarks (total 478)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.renderer = Renderer(width, height)

        # FPS tracking
        self._fps_time = time.time()
        self._fps_count = 0
        self._fps = 0.0

        print("[face-scanner] initialised — press Q to quit, S to screenshot")

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # read actual resolution (camera may not support requested size)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.width or actual_h != self.height:
            print(f"[face-scanner] camera gave {actual_w}×{actual_h} "
                  f"(requested {self.width}×{self.height}) — adapting")
            self.width  = actual_w
            self.height = actual_h
            self.renderer = Renderer(actual_w, actual_h)

        cv2.namedWindow("face-scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("face-scanner", self.width, self.height)

        shot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[face-scanner] camera read failed — exiting")
                break

            # mirror so it feels like a mirror
            frame = cv2.flip(frame, 1)

            # run MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.face_mesh.process(rgb)
            rgb.flags.writeable = True

            # extract data
            landmarks_3d, face_box, pose = self._extract(frame, results)

            # update FPS
            self._fps_count += 1
            now = time.time()
            if now - self._fps_time >= 0.5:
                self._fps = self._fps_count / (now - self._fps_time)
                self._fps_count = 0
                self._fps_time = now

            # render
            output = self.renderer.render(
                frame=frame,
                landmarks_3d=landmarks_3d,
                face_box=face_box,
                pose=pose,
                fps=self._fps,
                landmark_count=len(landmarks_3d),
            )

            cv2.imshow("face-scanner", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                shot_count += 1
                fname = f"screenshot_{shot_count:03d}.png"
                cv2.imwrite(fname, output)
                print(f"[face-scanner] saved {fname}")

        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

    # ── data extraction ────────────────────────────────────────────────────────

    def _extract(
        self,
        frame: np.ndarray,
        results,
    ) -> tuple[list, tuple | None, tuple[float, float, float]]:
        """
        Returns:
          landmarks_3d — list of (px, py, z_norm) for each visible landmark
          face_box     — (x, y, w, h) bounding rect or None
          pose         — (yaw, pitch, roll) degrees
        """
        if not results.multi_face_landmarks:
            return [], None, (0.0, 0.0, 0.0)

        face_lm = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # project landmarks to screen + normalize depth
        xs, ys, zs = [], [], []
        for lm in face_lm.landmark:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
            zs.append(lm.z)

        # normalize z to [0..1]: 0 = far, 1 = close (nose tip)
        z_arr = np.array(zs)
        z_norm = np.clip((z_arr - Z_FAR) / (Z_NEAR - Z_FAR), 0.0, 1.0)

        landmarks_3d = list(zip(xs, ys, z_norm.tolist()))

        # bounding box with padding
        pad = 30
        x_min = max(0, int(min(xs)) - pad)
        y_min = max(0, int(min(ys)) - pad)
        x_max = min(w, int(max(xs)) + pad)
        y_max = min(h, int(max(ys)) + pad)
        face_box = (x_min, y_min, x_max - x_min, y_max - y_min)

        # head pose
        pose = estimate_pose(face_lm, w, h)

        return landmarks_3d, face_box, pose
