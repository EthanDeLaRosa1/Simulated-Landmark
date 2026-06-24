"""
scanner.py — webcam capture loop + MediaPipe face mesh + rendering
Updated for mediapipe 0.10.30+
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import numpy as np
import time
import urllib.request
import os

from src.renderer import Renderer
from src.pose import estimate_pose


Z_NEAR = -0.15
Z_FAR  =  0.15

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[face-scanner] downloading face landmarker model (~29MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[face-scanner] model downloaded")


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

        download_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.renderer = Renderer(width, height)

        self._fps_time  = time.time()
        self._fps_count = 0
        self._fps       = 0.0

        print("[face-scanner] initialised — press Q to quit, S to screenshot")

    def run(self):
        cap = cv2.VideoCapture(self.cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.width or actual_h != self.height:
            print(f"[face-scanner] camera gave {actual_w}x{actual_h} — adapting")
            self.width  = actual_w
            self.height = actual_h
            self.renderer = Renderer(actual_w, actual_h)

        cv2.namedWindow("face-scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("face-scanner", self.width, self.height)

        shot_count  = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[face-scanner] camera read failed — exiting")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks_3d, face_box, pose = self._extract(frame, results)

            self._fps_count += 1
            now = time.time()
            if now - self._fps_time >= 0.5:
                self._fps       = self._fps_count / (now - self._fps_time)
                self._fps_count = 0
                self._fps_time  = now

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

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()

    def _extract(self, frame, results):
        if not results.face_landmarks:
            return [], None, (0.0, 0.0, 0.0)

        face_lm = results.face_landmarks[0]
        h, w = frame.shape[:2]

        xs, ys, zs = [], [], []
        for lm in face_lm:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
            zs.append(lm.z)

        z_arr  = np.array(zs)
        z_norm = np.clip((z_arr - Z_FAR) / (Z_NEAR - Z_FAR), 0.0, 1.0)

        landmarks_3d = list(zip(xs, ys, z_norm.tolist()))

        pad   = 30
        x_min = max(0, int(min(xs)) - pad)
        y_min = max(0, int(min(ys)) - pad)
        x_max = min(w, int(max(xs)) + pad)
        y_max = min(h, int(max(ys)) + pad)
        face_box = (x_min, y_min, x_max - x_min, y_max - y_min)

        # build a fake landmark list compatible with pose.py
        class _LM:
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
        class _LMList:
            def __init__(self, lms): self.landmark = lms

        lm_list = _LMList([_LM(lm.x, lm.y, lm.z) for lm in face_lm])
        pose = estimate_pose(lm_list, w, h)

        return landmarks_3d, face_box, pose