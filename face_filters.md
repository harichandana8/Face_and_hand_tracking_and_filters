# Face Filters (`face_filters.py`)

A real-time, Snapchat/Instagram-style face filter app built on OpenCV and
MediaPipe. Every filter graphic is **drawn in code** — there are no image
assets to download. Filters are switched with the **keyboard** and with
**hand gestures**.

The frame is mirrored for a natural selfie view.

## Filters

| # | Name | What it does |
| --- | --- | --- |
| 0 | None | No filter (passthrough). |
| 1 | Sunglasses | Tinted lenses with a bridge and glints, placed on the eyes, scaled by the eye distance and rotated to the head tilt. |
| 2 | Dog | Floppy ears above the forehead and a dog nose on the nose tip. A tongue appears when your mouth opens. |
| 3 | Makeup | Lipstick filled over the lips plus soft blush on the cheeks, feathered and blended. |
| 4 | Big Eyes | Local magnification around each eye using a radial image warp (uses iris landmarks). |
| 5 | Background Blur | Separates you from the background with selfie segmentation and blurs everything behind you. |
| 6 | Magic Hands | Sparkle particles stream from your fingertips. |

Face filters (1–4) support up to **2 faces** at once.

## Controls

### Keyboard

| Key | Action |
| --- | --- |
| `0`–`6` | Select that filter directly. |
| `n` / `N` / Right arrow | Next filter. |
| `p` / `P` / Left arrow | Previous filter. |
| `q` / `Q` / ESC | Quit (closing the window also quits). |

### Hand gestures

| Gesture | Action |
| --- | --- |
| **Pinch** (thumb tip to index tip) | Next filter. One pinch = one switch, with a ~1 second cooldown so it doesn't fire repeatedly. |
| **Open palm, held ~1 second** | Toggle filters off / back on. A progress ring shows the hold charging. |

When the filter changes, its name flashes briefly in the middle of the screen.

## Requirements

- Python 3.12 and the packages in [`requirements.txt`](./requirements.txt).
- A webcam.

MediaPipe is pinned to `0.10.21` because the app uses the legacy `mp.solutions`
API (`face_mesh`, `hands`, `selfie_segmentation`, `drawing_utils`), which was
removed in MediaPipe 0.10.30+.

## Running

From the project root, using the project virtual environment:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\python.exe face_filters.py
```

```bash
# macOS / Linux
.venv/bin/python face_filters.py
```

## Configuration

Named constants at the top of the file control the camera, thresholds, and the
look of each filter. A few of the most useful ones:

| Constant | Default | Meaning |
| --- | --- | --- |
| `CAMERA_INDEX` | `0` | Which webcam to open. |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `1280, 720` | Requested capture resolution. |
| `MAX_NUM_FACES` | `2` | Maximum faces for the face filters. |
| `MAX_NUM_HANDS` | `2` | Maximum hands tracked. |
| `DEFAULT_FILTER` | `1` | Filter active at startup. |
| `PINCH_CLOSE_RATIO` / `PINCH_OPEN_RATIO` | `0.25` / `0.40` | Pinch close/open thresholds, with hysteresis. |
| `PINCH_COOLDOWN_S` | `1.0` | Minimum seconds between pinch switches. |
| `PALM_HOLD_S` | `1.0` | Seconds an open palm must be held to toggle. |
| `LIP_COLOR`, `BLUSH_COLOR` | — | Makeup colors (BGR). |
| `BIG_EYE_RADIUS_RATIO`, `BIG_EYE_STRENGTH` | `0.14`, `0.4` | Size and intensity of the eye magnification. |
| `BG_BLUR_SIGMA` | `6` | Strength of the background blur. |

## How the code is organized

The app is structured so the per-frame work can run without a camera, which is
what the headless tests exercise.

- **`process_frame(frame_bgr, state, models, now=None)`** — the heart of the
  app. Runs the MediaPipe models, updates gestures, applies the active filter,
  and draws the HUD. Independent of the capture loop.
- **`main()`** — opens the camera, mirrors each frame, runs `process_frame`,
  handles keys, and releases the camera and windows on exit (`try/finally`).

### Graphics and placement

- **Static graphics** (sunglasses, ears, nose, tongue) are pre-rendered once at
  startup as transparent BGRA images (`render_*` functions, `build_assets`).
- They are placed each frame with a 2-point **similarity transform**
  (`place_asset` / `similarity_matrix`) so they scale and rotate with the face,
  and are safely clipped at frame edges.
- **Makeup, big eyes, and background blur** are computed directly from the
  landmarks each frame (`draw_makeup`, `magnify_region` + `apply_big_eyes`,
  `smooth_mask` + `blur_background`).

### Smoothing and gestures

- **`LandmarkSmoother`** applies exponential smoothing to face landmarks to
  reduce jitter, while snapping instantly on large jumps.
- **`GestureRecognizer`** is a pure state machine (no camera dependency) that
  turns hand landmarks into "next" and "toggle" events. It requires an open
  hand before arming a pinch and preserves the cooldown if tracking drops, so a
  single pinch can't accidentally fire twice.
- **`ParticleSystem`** powers the Magic Hands sparkles.
- **`key_to_action`** maps key codes (including Windows arrow-key codes via
  `cv2.waitKeyEx`) to actions.

## Verification status

- `python -m py_compile face_filters.py` passes.
- An 18-test headless suite passes, covering the gesture recognizer (pinch
  fires once per pinch, respects cooldown, survives tracking loss; open-palm
  hold toggles only after ~1s) and the keyboard mapping.
- A headless smoke test ran every filter (0–6) on the static image
  `other-files/img0.jpg`, saved a rendered PNG per filter, and those images
  were inspected to confirm the overlays land correctly (glasses on the eyes,
  dog nose on the nose, lipstick on the lips, eyes enlarged, background blurred
  while the person stays sharp).
- **Live webcam behavior and real-time FPS** were **not** verified
  automatically — run it with a camera to confirm.
