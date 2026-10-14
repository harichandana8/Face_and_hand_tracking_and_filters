# Face + Hand Tracking (`face_and_hand_Tracking.py`)

A real-time webcam demo that detects a face mesh and hand landmarks with
MediaPipe and draws them over the live video, along with an FPS counter.

## What it does

For every webcam frame the script:

1. Mirrors nothing — it shows the raw camera feed.
2. Runs MediaPipe **Hands** and **Face Mesh** on the frame.
3. Draws hand landmarks and their connections.
4. Draws the face mesh contours (outline of eyes, lips, eyebrows, face oval).
5. Overlays the current frames-per-second in the top-left corner.

It tracks up to **2 faces** and **2 hands** at once.

## Requirements

- Python 3.12 and the packages in [`requirements.txt`](./requirements.txt).
- A webcam.

MediaPipe is pinned to `0.10.21` because the script uses the legacy
`mp.solutions` API, which was removed in MediaPipe 0.10.30+.

## Running

From the project root, using the project virtual environment:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\python.exe face_and_hand_Tracking.py
```

```bash
# macOS / Linux
.venv/bin/python face_and_hand_Tracking.py
```

Press **`q`** or **ESC**, or close the window, to quit.

## Configuration

Settings live as named constants at the top of the file so you can tweak them
without touching the logic:

| Constant | Default | Meaning |
| --- | --- | --- |
| `CAMERA_INDEX` | `0` | Which webcam to open. |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `1920, 1080` | Requested capture resolution. The camera may pick a nearby size it actually supports. |
| `WINDOW_NAME` | `"Image"` | Title of the display window. |
| `MAX_NUM_FACES` | `2` | Maximum faces to track. |
| `MAX_NUM_HANDS` | `2` | Maximum hands to track. |
| `MIN_DETECTION_CONFIDENCE` | `0.5` | MediaPipe detection threshold. |
| `MIN_TRACKING_CONFIDENCE` | `0.5` | MediaPipe tracking threshold. |
| `QUIT_KEYS` | `{q, ESC}` | Keys that exit the program. |
| `FPS_*` | — | Position, font, scale, color, thickness of the FPS text. |

## How the code is organized

The file is split so detection/drawing can be tested without a camera.

- **`process_frame(frame_bgr, hands, face_mesh)`** — converts the frame to RGB,
  runs both MediaPipe models, and draws the hand and face annotations in place.
  This is the core logic and is independent of the webcam loop.
- **`draw_fps(frame_bgr, fps)`** — overlays the FPS counter.
- **`should_quit()`** — returns `True` when a quit key is pressed or the window
  is closed.
- **`main()`** — opens the camera, creates the MediaPipe models once as context
  managers, runs the capture loop, and guarantees the camera is released and
  windows are destroyed on exit (via `try/finally`).

### Notable robustness details

- The frame is converted **BGR → RGB** before going to MediaPipe (OpenCV uses
  BGR; MediaPipe expects RGB), and marked read-only during processing to let
  MediaPipe skip an internal copy.
- A failed `cap.read()` prints a message and exits cleanly instead of crashing.
- The FPS calculation guards against a zero time interval (no divide-by-zero).
- Face connections use `FACEMESH_CONTOURS`; the older `FACE_CONNECTIONS`
  constant no longer exists in MediaPipe 0.10.21.

## Verification status

- `python -m py_compile face_and_hand_Tracking.py` passes.
- A headless smoke test ran `process_frame` and `draw_fps` on the static image
  `other-files/img0.jpg` with no window and no exceptions.
- **Live webcam behavior** (the capture loop, window, and quit handling) was
  **not** verified automatically — run it with a camera to confirm.
