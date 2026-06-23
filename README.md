# face-scanner

A holographic face point cloud visualizer built on MediaPipe Face Mesh.  
468 depth-colored landmarks, glow trails, scanline sweep, head pose HUD — runs on any webcam.

---

## setup

```bash
# clone or copy this folder, then:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Tested on Python 3.10–3.12. A Logitech C920 or similar USB webcam works well.

---

## controls

| key | action |
|-----|--------|
| `Q` | quit |
| `S` | save screenshot to `screenshot_NNN.png` |

---

## what you see

| element | description |
|---------|-------------|
| **point cloud** | 468 (or 478 with iris) face landmarks, colored by depth |
| **depth color** | blue = far from camera, green = mid, amber/coral = close (nose tip) |
| **glow trails** | each point leaves a decaying light bloom that fades over ~20 frames |
| **scanline** | green sweep line that bounces top to bottom |
| **corner brackets** | pulsing bounding box around detected face |
| **HUD top-left** | FPS, landmark count, elapsed time |
| **HUD top-right** | yaw / pitch / roll head pose angles |
| **depth legend** | color bar at bottom-left |

---

## architecture

```
main.py               entry point
src/
  scanner.py          webcam loop, MediaPipe, data extraction
  renderer.py         all visual effects (points, glow, scanline, HUD, vignette)
  pose.py             yaw/pitch/roll from 6-point PnP
```

### how depth works

MediaPipe returns each landmark with a `z` value representing depth relative  
to the face center. The nose tip is approximately `z = -0.12` (closest to the  
camera) and the ears/temples are around `z = +0.12` (furthest). We normalize  
this range to `[0..1]` and map it through a 5-stop color gradient:

```
0.0 (far)  →  deep blue
0.25       →  cyan
0.5        →  green
0.75       →  amber
1.0 (near) →  coral/red
```

### head pose

Six canonical face points (nose tip, chin, eye corners, mouth corners) are  
passed to OpenCV's `solvePnP` against a known 3D face model. The resulting  
rotation matrix is decomposed into Euler angles (yaw, pitch, roll).

---

## ideas to extend

### visual
- `renderer.py` → add mesh triangles using `mp.solutions.face_mesh.FACEMESH_TESSELATION`
- Add a second render mode toggled by `M` key (wireframe vs points)
- Make glow color shift with head pose — e.g. yaw → hue rotation
- Export a point cloud to `.ply` format for 3D software

### data / application
- Log pose angles to CSV for downstream analysis
- Trigger events on pose thresholds (e.g. "looking away")
- Add attention zone overlay (quadrant the user is looking toward)
- Pipe pose data via OSC to TouchDesigner or Resolume for live projection mapping
- Add PERCLOS (eye-openness ratio) for drowsiness detection

### projection mapping bridge
```python
# send head position over UDP to a projection tool
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(f"{yaw},{pitch},{roll}".encode(), ("127.0.0.1", 7000))
```

---

## troubleshooting

**camera not opening** — try `camera_index=1` or `camera_index=2` in `scanner.py`  
**low FPS** — reduce resolution: change `width=640, height=480` in `FaceScanner()`  
**no face detected** — improve lighting, face the camera straight on to initialise  
**import error** — make sure you activated the venv before running
