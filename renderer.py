"""
renderer.py — all visual effects for the holographic face scanner
"""

import cv2
import numpy as np
import math
import time


# ── color palette ──────────────────────────────────────────────────────────────
# depth gradient: deep blue (far) → cyan → green → amber → coral (close)
DEPTH_PALETTE = [
    (180, 60,  20),   # far  — deep blue (BGR)
    (200, 140, 20),   # mid  — cyan
    (60,  180, 60),   # mid  — green
    (20,  180, 220),  # near — amber
    (40,  80,  240),  # very near — coral/red
]

HUD_GREEN    = (80, 220, 80)
HUD_CYAN     = (220, 220, 60)
HUD_DIM      = (40,  100, 40)
SCANLINE_COL = (0,   180, 80)
GLOW_COL     = (0,   255, 120)


def depth_color(z_norm: float) -> tuple[int, int, int]:
    """Map a normalized z value [0..1] (0=far, 1=close) to a BGR color."""
    z_norm = max(0.0, min(1.0, z_norm))
    n = len(DEPTH_PALETTE) - 1
    idx = z_norm * n
    lo = int(idx)
    hi = min(lo + 1, n)
    t = idx - lo
    b = int(DEPTH_PALETTE[lo][0] * (1 - t) + DEPTH_PALETTE[hi][0] * t)
    g = int(DEPTH_PALETTE[lo][1] * (1 - t) + DEPTH_PALETTE[hi][1] * t)
    r = int(DEPTH_PALETTE[lo][2] * (1 - t) + DEPTH_PALETTE[hi][2] * t)
    return (b, g, r)


class Renderer:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.start_time = time.time()

        # scanline state
        self.scan_y = 0
        self.scan_speed = 3          # px per frame
        self.scan_dir = 1

        # persistent glow layer (decays each frame)
        self.glow_layer = np.zeros((height, width, 3), dtype=np.float32)
        self.glow_decay = 0.72       # how fast trails fade

        # corner bracket lengths
        self.bracket = 28

    # ── public API ─────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        landmarks_3d: list[tuple[float, float, float]],
        face_box: tuple[int, int, int, int] | None,
        pose: tuple[float, float, float],
        fps: float,
        landmark_count: int,
    ) -> np.ndarray:
        """
        Compose a full holographic frame.
        landmarks_3d: list of (px, py, z_norm) — screen pixels + normalized depth
        face_box: (x, y, w, h) bounding box or None
        pose: (yaw, pitch, roll) in degrees
        """
        # 1. dark tinted background from webcam
        canvas = self._make_background(frame)

        # 2. update scanline position
        self._tick_scanline()

        # 3. draw scanline glow sweep
        self._draw_scanline(canvas)

        # 4. draw grid overlay
        self._draw_grid(canvas)

        # 5. draw point cloud with glow trails
        if landmarks_3d:
            self._draw_points(canvas, landmarks_3d)

        # 6. face bounding box with corner brackets
        if face_box is not None:
            self._draw_face_box(canvas, face_box)

        # 7. HUD: stats, pose, title
        self._draw_hud(canvas, pose, fps, landmark_count)

        # 8. vignette
        self._draw_vignette(canvas)

        return canvas

    # ── background ─────────────────────────────────────────────────────────────

    def _make_background(self, frame: np.ndarray) -> np.ndarray:
        """Darken and tint the webcam frame greenish for the holo look."""
        dark = (frame.astype(np.float32) * 0.18).astype(np.uint8)
        tinted = dark.copy()
        tinted[:, :, 1] = np.clip(dark[:, :, 1].astype(np.int16) + 12, 0, 255).astype(np.uint8)
        return tinted

    # ── scanline ───────────────────────────────────────────────────────────────

    def _tick_scanline(self):
        self.scan_y += self.scan_speed * self.scan_dir
        if self.scan_y >= self.h:
            self.scan_dir = -1
        elif self.scan_y <= 0:
            self.scan_dir = 1
        self.scan_y = max(0, min(self.h - 1, self.scan_y))

    def _draw_scanline(self, canvas: np.ndarray):
        """A horizontal glow line that sweeps up and down."""
        for dy, alpha in [(-2, 0.04), (-1, 0.12), (0, 0.55), (1, 0.12), (2, 0.04)]:
            y = self.scan_y + dy
            if 0 <= y < self.h:
                row = canvas[y].astype(np.float32)
                row[:, 1] = np.clip(row[:, 1] + 255 * alpha, 0, 255)
                row[:, 0] = np.clip(row[:, 0] + 120 * alpha, 0, 255)
                canvas[y] = row.astype(np.uint8)

    # ── grid ───────────────────────────────────────────────────────────────────

    def _draw_grid(self, canvas: np.ndarray):
        """Subtle perspective grid lines."""
        t = time.time() - self.start_time
        # horizontal lines — fixed spacing
        for y in range(0, self.h, 60):
            alpha = 0.18 if (y % 180 == 0) else 0.06
            cv2.line(canvas, (0, y), (self.w, y), HUD_DIM, 1,
                     lineType=cv2.LINE_AA)
        # vertical lines
        for x in range(0, self.w, 80):
            cv2.line(canvas, (x, 0), (x, self.h), HUD_DIM, 1,
                     lineType=cv2.LINE_AA)

    # ── points ─────────────────────────────────────────────────────────────────

    def _draw_points(
        self,
        canvas: np.ndarray,
        landmarks: list[tuple[float, float, float]],
    ):
        """Draw depth-colored dots with glow decay trails."""
        # decay previous glow layer
        self.glow_layer *= self.glow_decay

        # paint current points onto glow layer and canvas
        for px, py, z_norm in landmarks:
            x, y = int(px), int(py)
            if not (0 <= x < self.w and 0 <= y < self.h):
                continue

            color = depth_color(z_norm)
            radius = int(2 + z_norm * 2.5)  # closer = slightly bigger

            # glow trail (written to float layer)
            r = radius + 3
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(self.w, x + r + 1), min(self.h, y + r + 1)
            for gy in range(y0, y1):
                for gx in range(x0, x1):
                    dist = math.sqrt((gx - x) ** 2 + (gy - y) ** 2)
                    if dist < r:
                        falloff = (1 - dist / r) ** 2
                        self.glow_layer[gy, gx, 0] = min(255, self.glow_layer[gy, gx, 0] + color[0] * falloff * 0.6)
                        self.glow_layer[gy, gx, 1] = min(255, self.glow_layer[gy, gx, 1] + color[1] * falloff * 0.6)
                        self.glow_layer[gy, gx, 2] = min(255, self.glow_layer[gy, gx, 2] + color[2] * falloff * 0.6)

            # sharp dot on canvas
            cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            # bright core
            cv2.circle(canvas, (x, y), max(1, radius - 1),
                       tuple(min(255, int(c * 1.6)) for c in color), -1,
                       lineType=cv2.LINE_AA)

        # blend glow layer onto canvas
        glow_uint8 = np.clip(self.glow_layer, 0, 255).astype(np.uint8)
        canvas[:] = np.clip(
            canvas.astype(np.float32) + glow_uint8.astype(np.float32) * 0.4,
            0, 255
        ).astype(np.uint8)

    # ── face box ───────────────────────────────────────────────────────────────

    def _draw_face_box(self, canvas: np.ndarray, box: tuple[int, int, int, int]):
        """Corner bracket style bounding box — no full rectangle."""
        x, y, w, h = box
        b = self.bracket
        col = HUD_GREEN
        t = 2

        # pulse: alpha flicker on the bracket
        pulse = 0.7 + 0.3 * math.sin(time.time() * 4)
        col = tuple(int(c * pulse) for c in col)

        corners = [
            # top-left
            [(x, y + b), (x, y), (x + b, y)],
            # top-right
            [(x + w - b, y), (x + w, y), (x + w, y + b)],
            # bottom-left
            [(x, y + h - b), (x, y + h), (x + b, y + h)],
            # bottom-right
            [(x + w - b, y + h), (x + w, y + h), (x + w, y + h - b)],
        ]
        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], col, t, lineType=cv2.LINE_AA)

        # center crosshair
        cx, cy = x + w // 2, y + h // 2
        arm = 10
        cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), col, 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), col, 1, lineType=cv2.LINE_AA)

        # label above box
        label = "FACE DETECTED"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = x + w // 2 - tw // 2
        ly = y - 8
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # ── HUD ────────────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        canvas: np.ndarray,
        pose: tuple[float, float, float],
        fps: float,
        landmark_count: int,
    ):
        yaw, pitch, roll = pose
        t = time.time() - self.start_time
        elapsed = f"{int(t // 60):02d}:{int(t % 60):02d}"

        # ── top-left block ──
        lines_tl = [
            ("FACE SCANNER v1.0",   HUD_GREEN,  0.45, 1),
            (f"FPS  {fps:05.1f}",   HUD_CYAN,   0.38, 1),
            (f"PTS  {landmark_count:04d}", HUD_CYAN, 0.38, 1),
            (f"TIME {elapsed}",     HUD_DIM,    0.35, 1),
        ]
        y_off = 22
        for text, col, scale, thick in lines_tl:
            cv2.putText(canvas, text, (12, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
            y_off += 20

        # ── top-right: pose angles ──
        pose_lines = [
            f"YAW   {yaw:+.1f}",
            f"PITCH {pitch:+.1f}",
            f"ROLL  {roll:+.1f}",
        ]
        y_off = 22
        for line in pose_lines:
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(canvas, line, (self.w - tw - 12, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_CYAN, 1, cv2.LINE_AA)
            y_off += 20

        # ── bottom-left: depth legend ──
        legend_y = self.h - 14
        cv2.putText(canvas, "DEPTH", (12, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_DIM, 1, cv2.LINE_AA)
        bar_x = 60
        bar_w = 120
        for i in range(bar_w):
            col = depth_color(i / bar_w)
            cv2.line(canvas, (bar_x + i, legend_y - 6),
                     (bar_x + i, legend_y - 1), col, 1)
        cv2.putText(canvas, "FAR", (bar_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, HUD_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, "CLOSE", (bar_x + bar_w - 28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, HUD_DIM, 1, cv2.LINE_AA)

        # ── bottom-right: mode indicator ──
        mode = "[ POINT CLOUD ]"
        (tw, _), _ = cv2.getTextSize(mode, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        pulse_col = tuple(int(c * (0.6 + 0.4 * math.sin(t * 2))) for c in HUD_GREEN)
        cv2.putText(canvas, mode, (self.w - tw - 12, self.h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, pulse_col, 1, cv2.LINE_AA)

        # ── thin border frame ──
        cv2.rectangle(canvas, (2, 2), (self.w - 3, self.h - 3), HUD_DIM, 1)

    # ── vignette ───────────────────────────────────────────────────────────────

    def _draw_vignette(self, canvas: np.ndarray):
        """Darken edges for cinematic depth."""
        if not hasattr(self, '_vignette'):
            vig = np.zeros((self.h, self.w), dtype=np.float32)
            cx, cy = self.w / 2, self.h / 2
            max_d = math.sqrt(cx ** 2 + cy ** 2)
            for y in range(self.h):
                for x in range(self.w):
                    d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_d
                    vig[y, x] = max(0.0, 1.0 - d * 1.2)
            self._vignette = vig[:, :, np.newaxis]

        canvas[:] = np.clip(
            canvas.astype(np.float32) * self._vignette, 0, 255
        ).astype(np.uint8)
