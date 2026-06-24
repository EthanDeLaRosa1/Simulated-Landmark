"""
renderer.py — minimal scientific aesthetic
Reference: RealSense Viewer, CloudCompare, medical depth imaging
"""

import cv2
import numpy as np
import math
import time


def depth_color(z_norm: float) -> tuple:
    """
    Jet colormap — the actual standard in depth/thermal imaging research.
    0 = far (dark blue), 1 = close (dark red)
    """
    z = max(0.0, min(1.0, z_norm))
    if z < 0.25:
        t = z / 0.25
        return (int(128 + 127*t), 0, 0)           # dark blue → blue
    elif z < 0.5:
        t = (z - 0.25) / 0.25
        return (255, int(255*t), 0)                # blue → cyan
    elif z < 0.75:
        t = (z - 0.5) / 0.25
        return (int(255*(1-t)), 255, int(255*t))   # cyan → yellow
    else:
        t = (z - 0.75) / 0.25
        return (0, int(255*(1-t)), 255)            # yellow → red


FONT        = cv2.FONT_HERSHEY_PLAIN
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN
C_WHITE     = (210, 215, 210)
C_DIM       = ( 90,  95,  90)
C_DIMMER    = ( 45,  48,  45)
C_ACCENT    = (130, 180, 130)   # single muted green for interactive elements only
C_BG        = (  8,  10,   8)


class Renderer:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.start_time = time.time()
        self.frame_n    = 0
        self._vignette  = self._build_vignette()

    # ── public ────────────────────────────────────────────────────────────────

    def render(self, frame, landmarks_3d, face_box, pose, fps, landmark_count):
        self.frame_n += 1

        # pure black canvas — no camera feed, no grid, no decoration
        canvas = np.full((self.h, self.w, 3), C_BG, dtype=np.uint8)

        if landmarks_3d:
            self._draw_points(canvas, landmarks_3d)

        if face_box:
            self._draw_box(canvas, face_box)

        self._draw_hud(canvas, pose, fps, landmark_count)

        # soft edge darkening only
        canvas = (canvas.astype(np.float32) * self._vignette).clip(0,255).astype(np.uint8)

        return canvas

    # ── points ────────────────────────────────────────────────────────────────

    def _draw_points(self, canvas, landmarks):
        # sort far → near so close points paint over distant ones
        pts = sorted(landmarks, key=lambda p: p[2])

        for px, py, z_norm in pts:
            x, y = int(px), int(py)
            if not (0 <= x < self.w and 0 <= y < self.h):
                continue
            col = depth_color(z_norm)
            # single pixel for distant points, 2px for close — that's it
            r = 1 if z_norm < 0.55 else 2
            cv2.circle(canvas, (x, y), r, col, -1, cv2.LINE_AA)

    # ── bounding box ──────────────────────────────────────────────────────────

    def _draw_box(self, canvas, box):
        x, y, w, h = box
        # single thin rectangle, dim — just enough to show the region
        cv2.rectangle(canvas, (x, y), (x+w, y+h), C_DIMMER, 1)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, canvas, pose, fps, landmark_count):
        yaw, pitch, roll = pose
        t = time.time() - self.start_time

        # ── left column ──────────────────────────────────────────────────────
        lx = 12
        rows = [
            (f"fps    {fps:5.1f}",          C_DIM),
            (f"pts    {landmark_count:4d}", C_DIM),
            (f"t      {t:6.1f}s",           C_DIMMER),
        ]
        for i, (text, col) in enumerate(rows):
            cv2.putText(canvas, text, (lx, 18 + i*16),
                        FONT_MONO, 1.0, col, 1, cv2.LINE_AA)

        # ── right column: pose ───────────────────────────────────────────────
        rx = self.w - 148
        cv2.putText(canvas, "head pose", (rx, 18),
                    FONT_MONO, 1.0, C_DIMMER, 1, cv2.LINE_AA)

        for i, (label, val) in enumerate([("yaw", yaw), ("pitch", pitch), ("roll", roll)]):
            y_off = 34 + i * 16
            text  = f"{label:<6}{val:+6.1f}"
            col   = C_WHITE if abs(val) > 15 else C_DIM
            cv2.putText(canvas, text, (rx, y_off),
                        FONT_MONO, 1.0, col, 1, cv2.LINE_AA)

        # ── depth scale — bottom left, minimal ───────────────────────────────
        bx, by = 12, self.h - 18
        cv2.putText(canvas, "depth", (bx, by),
                    FONT_MONO, 0.85, C_DIMMER, 1, cv2.LINE_AA)
        sx = bx + 42
        for i in range(100):
            col = depth_color(i / 100)
            # dim the bar down — it's reference, not decoration
            col = tuple(int(c * 0.55) for c in col)
            cv2.line(canvas, (sx+i, by-8), (sx+i, by-3), col, 1)
        cv2.putText(canvas, "far",  (sx,       by+2), FONT_MONO, 0.7, C_DIMMER, 1, cv2.LINE_AA)
        cv2.putText(canvas, "near", (sx+76, by+2), FONT_MONO, 0.7, C_DIMMER, 1, cv2.LINE_AA)

        # ── single pixel border ───────────────────────────────────────────────
        cv2.rectangle(canvas, (0,0), (self.w-1, self.h-1), C_DIMMER, 1)

    # ── vignette ──────────────────────────────────────────────────────────────

    def _build_vignette(self):
        cx, cy = self.w/2, self.h/2
        ys, xs = np.mgrid[0:self.h, 0:self.w]
        d = np.sqrt(((xs-cx)/cx)**2 + ((ys-cy)/cy)**2)
        v = np.clip(1.0 - d * 0.4, 0.0, 1.0)
        return v[:,:,np.newaxis].astype(np.float32)