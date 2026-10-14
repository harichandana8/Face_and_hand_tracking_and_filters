"""Real-time, procedurally drawn face and hand filters.

Controls: 0-6 selects a filter, N/P or Left/Right cycles filters, a thumb/index
pinch advances, and holding an open palm for one second toggles filters off/on.
Press Q or ESC (or close the window) to quit. Frames are mirrored for selfie use.

VERIFICATION: See .agents/tasks/verification.md for the exact compile, headless
smoke, visual inspection, gesture/key test results, and unverified live behavior.
"""

import contextlib
import dataclasses
import math
import time
import typing

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------- config ---
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
WINDOW_NAME = "Face Filters"
MAX_NUM_FACES = 2
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
SEGMENTATION_MODEL = 1
DEFAULT_FILTER = 1
FILTER_NAMES = [
    "None", "Sunglasses", "Dog", "Makeup", "Big Eyes",
    "Background Blur", "Magic Hands",
]
FACE_FILTERS = {1, 2, 3, 4}
FILTER_BG_BLUR = 5
FILTER_MAGIC = 6
LANDMARK_EMA_ALPHA = 0.5
LANDMARK_SNAP_RATIO = 0.25
PINCH_CLOSE_RATIO = 0.25
PINCH_OPEN_RATIO = 0.40
PINCH_MIN_INDEX_REACH = 1.0
PINCH_COOLDOWN_S = 1.0
PALM_HOLD_S = 1.0
FINGER_EXTENDED_RATIO = 1.1
THUMB_OUT_RATIO = 0.45
MOUTH_OPEN_RATIO = 0.05
BIG_EYE_RADIUS_RATIO = 0.14
BIG_EYE_STRENGTH = 0.4
LIP_COLOR = (70, 30, 190)
LIP_STRENGTH = 0.55
BLUSH_COLOR = (140, 120, 240)
BLUSH_STRENGTH = 0.35
BLUSH_RADIUS_RATIO = 0.10
SEG_FEATHER_SIGMA = 5
SEG_REFINE_LOW = 0.3
SEG_REFINE_SPAN = 0.4
SEG_MASK_EMA = 0.6
BG_BLUR_SIGMA = 6
BG_BLUR_DOWNSCALE = 4
PARTICLES_PER_TIP = 3
MAX_PARTICLES = 800
PARTICLE_LIFE_S = (0.6, 1.0)
FLASH_S = 0.8
FPS_EMA = 0.9
KEY_ESC = 27
KEY_RIGHT = (2555904, 65363)
KEY_LEFT = (2424832, 65361)

# ------------------------------------------------------ mediapipe handles ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation
mp_draw = mp.solutions.drawing_utils

# -------------------------------------------------------- landmark indices ---
NOSE_TIP, FOREHEAD, CHIN = 1, 10, 152
FACE_LEFT_EDGE, FACE_RIGHT_EDGE = 234, 454
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
IRIS_CENTERS = (468, 473)
UPPER_INNER, LOWER_INNER = 13, 14
MOUTH_CORNERS = (61, 291)
CHEEKS = (50, 280)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
              270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
              310, 311, 312, 13, 82, 81, 80, 191]
WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)
ALL_TIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (0, 5, 9, 13, 17)


@dataclasses.dataclass
class Asset:
    """A BGRA graphic and its two placement anchors."""

    image: np.ndarray
    anchor_a: tuple[float, float]
    anchor_b: tuple[float, float]


@dataclasses.dataclass
class Models:
    """Long-lived MediaPipe model instances."""

    face_mesh: object
    hands: object
    segmenter: object


def similarity_matrix(src_a, src_b, dst_a, dst_b):
    """Return a 2-point similarity transform from source to destination."""
    sa, sb = np.asarray(src_a, np.float64), np.asarray(src_b, np.float64)
    da, db = np.asarray(dst_a, np.float64), np.asarray(dst_b, np.float64)
    source = complex(*(sb - sa))
    if abs(source) < 1e-8:
        raise ValueError("Asset anchors must be distinct")
    scale_rotation = complex(*(db - da)) / source
    a, b = scale_rotation.real, scale_rotation.imag
    rotation = np.array([[a, -b], [b, a]], np.float64)
    translation = da - rotation @ sa
    return np.column_stack((rotation, translation)).astype(np.float32)


def alpha_composite(destination, overlay):
    """Composite a same-size BGRA overlay into a BGR destination in place."""
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    mixed = destination.astype(np.float32) * (1.0 - alpha)
    mixed += overlay[..., :3].astype(np.float32) * alpha
    destination[:] = np.clip(mixed, 0, 255).astype(np.uint8)


def place_asset(frame, asset, dst_a, dst_b):
    """Warp and composite an asset, safely clipping it to frame boundaries."""
    points = np.asarray((dst_a, dst_b), np.float64)
    if not np.all(np.isfinite(points)):
        return False
    try:
        matrix = similarity_matrix(asset.anchor_a, asset.anchor_b, dst_a, dst_b)
    except ValueError:
        return False
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    if not math.isfinite(scale) or scale <= 0 or scale > 20:
        return False
    height, width = asset.image.shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                       np.float32)
    transformed = cv2.transform(corners[None, ...], matrix)[0]
    x0 = max(0, int(math.floor(float(transformed[:, 0].min()))))
    y0 = max(0, int(math.floor(float(transformed[:, 1].min()))))
    x1 = min(frame.shape[1], int(math.ceil(float(transformed[:, 0].max()))))
    y1 = min(frame.shape[0], int(math.ceil(float(transformed[:, 1].max()))))
    if x0 >= x1 or y0 >= y1:
        return False
    local_matrix = matrix.copy()
    local_matrix[:, 2] -= (x0, y0)
    warped = cv2.warpAffine(
        asset.image, local_matrix, (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha_composite(frame[y0:y1, x0:x1], warped)
    return True


def render_sunglasses():
    """Render tinted sunglasses as a reusable BGRA image."""
    image = np.zeros((220, 640, 4), np.uint8)
    lens_mask = np.zeros(image.shape[:2], np.uint8)
    for center in ((170, 105), (470, 105)):
        cv2.ellipse(lens_mask, center, (125, 85), 0, 0, 360, 255, -1)
    gradient = np.linspace(235, 170, image.shape[0], dtype=np.uint8)[:, None]
    image[..., :3][lens_mask > 0] = (40, 25, 15)
    image[..., 3] = np.where(lens_mask > 0, gradient, 0).astype(np.uint8)
    for center in ((170, 105), (470, 105)):
        cv2.ellipse(image, center, (125, 85), 0, 0, 360, (15, 15, 15, 255), 12)
        cv2.ellipse(image, (center[0] - 38, center[1] - 27), (25, 10), -25,
                    0, 360, (255, 255, 255, 175), -1)
    cv2.ellipse(image, (320, 105), (30, 20), 180, 180, 360,
                (15, 15, 15, 255), 12)
    cv2.line(image, (35, 85), (5, 65), (15, 15, 15, 255), 12)
    cv2.line(image, (605, 85), (635, 65), (15, 15, 15, 255), 12)
    return Asset(image, (170, 105), (470, 105))


def render_dog_ear(flipped=False):
    """Render one floppy dog ear."""
    image = np.zeros((340, 220, 4), np.uint8)
    cv2.ellipse(image, (108, 170), (78, 145), 12, 0, 360,
                (45, 85, 145, 255), -1)
    cv2.ellipse(image, (108, 175), (48, 110), 12, 0, 360,
                (65, 120, 190, 235), -1)
    cv2.ellipse(image, (108, 170), (78, 145), 12, 0, 360,
                (30, 55, 85, 255), 8)
    if flipped:
        image = cv2.flip(image, 1)
    return Asset(image, (50, 40), (170, 40))


def render_dog_nose():
    """Render a glossy dog nose."""
    image = np.zeros((150, 200, 4), np.uint8)
    cv2.ellipse(image, (100, 72), (82, 58), 0, 0, 360,
                (18, 18, 22, 255), -1)
    cv2.ellipse(image, (73, 53), (25, 13), -20, 0, 360,
                (125, 125, 135, 190), -1)
    cv2.ellipse(image, (65, 82), (13, 8), 0, 0, 360, (2, 2, 3, 255), -1)
    cv2.ellipse(image, (135, 82), (13, 8), 0, 0, 360, (2, 2, 3, 255), -1)
    return Asset(image, (15, 70), (185, 70))


def render_tongue():
    """Render a rounded pink tongue."""
    image = np.zeros((200, 140, 4), np.uint8)
    cv2.ellipse(image, (70, 95), (54, 90), 0, 0, 180,
                (125, 100, 245, 250), -1)
    cv2.rectangle(image, (16, 20), (124, 98), (125, 100, 245, 250), -1)
    cv2.line(image, (70, 65), (70, 175), (105, 65, 205, 220), 5)
    return Asset(image, (15, 20), (125, 20))


def build_assets():
    """Pre-render every static filter graphic once."""
    return {
        "sunglasses": render_sunglasses(),
        "ear_left": render_dog_ear(),
        "ear_right": render_dog_ear(True),
        "dog_nose": render_dog_nose(),
        "tongue": render_tongue(),
    }


def face_axes(points):
    """Return face-horizontal/down axes and face width."""
    left = points[list(LEFT_EYE_CORNERS)].mean(axis=0)
    right = points[list(RIGHT_EYE_CORNERS)].mean(axis=0)
    delta = right - left
    length = float(np.linalg.norm(delta))
    unit = delta / length if length > 1e-6 else np.array([1.0, 0.0])
    down = np.array([-unit[1], unit[0]], np.float32)
    width = float(np.linalg.norm(points[FACE_RIGHT_EDGE] - points[FACE_LEFT_EDGE]))
    return unit.astype(np.float32), down, max(width, 1.0)


def mouth_open_ratio(points):
    """Measure inner-lip separation relative to face height."""
    height = float(np.linalg.norm(points[FOREHEAD] - points[CHIN]))
    return float(np.linalg.norm(points[UPPER_INNER] - points[LOWER_INNER])) / max(height, 1.0)


def draw_sunglasses(frame, points, assets):
    """Place sunglasses over one face."""
    left = points[list(LEFT_EYE_CORNERS)].mean(axis=0)
    right = points[list(RIGHT_EYE_CORNERS)].mean(axis=0)
    place_asset(frame, assets["sunglasses"], left, right)


def draw_dog(frame, points, assets):
    """Place floppy ears, nose, and an optional tongue on one face."""
    unit, down, width = face_axes(points)
    forehead = points[FOREHEAD]
    mouth_width = float(np.linalg.norm(points[MOUTH_CORNERS[0]] - points[MOUTH_CORNERS[1]]))
    if mouth_open_ratio(points) > MOUTH_OPEN_RATIO:
        center = points[LOWER_INNER]
        place_asset(frame, assets["tongue"], center - 0.35 * mouth_width * unit,
                    center + 0.35 * mouth_width * unit)
    place_asset(frame, assets["ear_left"],
                forehead - 0.62 * width * unit - 0.10 * width * down,
                forehead - 0.30 * width * unit - 0.10 * width * down)
    place_asset(frame, assets["ear_right"],
                forehead + 0.30 * width * unit - 0.10 * width * down,
                forehead + 0.62 * width * unit - 0.10 * width * down)
    nose = points[NOSE_TIP]
    place_asset(frame, assets["dog_nose"], nose - 0.17 * width * unit,
                nose + 0.17 * width * unit)


def tint_with_mask(region, mask, color, strength):
    """Tint a BGR region according to a soft uint8 or float mask."""
    weight = mask.astype(np.float32)
    if weight.max(initial=0) > 1.0:
        weight /= 255.0
    weight = np.clip(weight * strength, 0.0, 1.0)[..., None]
    result = region.astype(np.float32) * (1.0 - weight)
    result += np.asarray(color, np.float32) * weight
    region[:] = np.clip(result, 0, 255).astype(np.uint8)


def draw_makeup(frame, points):
    """Apply feathered lipstick and blush to one face."""
    _, _, width = face_axes(points)
    margin = int(0.1 * width)
    all_x, all_y = points[:, 0], points[:, 1]
    x0 = max(0, int(np.floor(all_x.min())) - margin)
    y0 = max(0, int(np.floor(all_y.min())) - margin)
    x1 = min(frame.shape[1], int(np.ceil(all_x.max())) + margin + 1)
    y1 = min(frame.shape[0], int(np.ceil(all_y.max())) + margin + 1)
    if x0 >= x1 or y0 >= y1:
        return
    region = frame[y0:y1, x0:x1]
    offset = np.array([x0, y0], np.float32)
    lip_mask = np.zeros(region.shape[:2], np.uint8)
    outer = np.rint(points[LIPS_OUTER] - offset).astype(np.int32)
    inner = np.rint(points[LIPS_INNER] - offset).astype(np.int32)
    cv2.fillPoly(lip_mask, [outer], 255)
    cv2.fillPoly(lip_mask, [inner], 0)
    lip_mask = cv2.GaussianBlur(lip_mask, (0, 0), max(1.0, 0.01 * width))
    tint_with_mask(region, lip_mask, LIP_COLOR, LIP_STRENGTH)

    blush_mask = np.zeros(region.shape[:2], np.uint8)
    radius = max(2, int(BLUSH_RADIUS_RATIO * width))
    for index in CHEEKS:
        center = tuple(np.rint(points[index] - offset).astype(int))
        cv2.circle(blush_mask, center, radius, 255, -1)
    blush_mask = cv2.GaussianBlur(blush_mask, (0, 0), max(1.0, 0.6 * radius))
    tint_with_mask(region, blush_mask, BLUSH_COLOR, BLUSH_STRENGTH)


def magnify_region(frame, center, radius, strength=BIG_EYE_STRENGTH):
    """Magnify a circular region with a smooth radial remap."""
    if radius <= 1 or strength == 0:
        return
    cx, cy = float(center[0]), float(center[1])
    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1 = min(frame.shape[1], int(cx + radius) + 1)
    y1 = min(frame.shape[0], int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    distance = np.sqrt(dx * dx + dy * dy)
    t = np.clip(distance / radius, 0.0, 1.0)
    factor = 1.0 - strength * (1.0 - t) ** 2
    map_x = (cx + dx * factor - x0).astype(np.float32)
    map_y = (cy + dy * factor - y0).astype(np.float32)
    source = frame[y0:y1, x0:x1].copy()
    remapped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    inside = distance <= radius
    frame[y0:y1, x0:x1][inside] = remapped[inside]


def apply_big_eyes(frame, points):
    """Magnify both iris regions for one face."""
    _, _, width = face_axes(points)
    if len(points) > max(IRIS_CENTERS):
        centers = (points[IRIS_CENTERS[0]], points[IRIS_CENTERS[1]])
    else:
        centers = (points[list(LEFT_EYE_CORNERS)].mean(axis=0),
                   points[list(RIGHT_EYE_CORNERS)].mean(axis=0))
    for center in centers:
        magnify_region(frame, center, BIG_EYE_RADIUS_RATIO * width)


def smooth_mask(previous, raw):
    """Contrast-refine, feather, and temporally smooth a segmentation mask.

    MediaPipe's raw confidence stays mid-range over thin foreground limbs such
    as raised hands, so feathering it directly leaves those regions blended
    with the blurred background. Remapping confidence to push likely
    foreground toward 1 before feathering keeps hands and arms sharp.
    """
    confidence = np.asarray(raw, np.float32)
    refined = np.clip((confidence - SEG_REFINE_LOW) / SEG_REFINE_SPAN, 0.0, 1.0)
    current = cv2.GaussianBlur(refined, (0, 0), SEG_FEATHER_SIGMA)
    current = np.clip(current, 0.0, 1.0)
    if previous is not None and previous.shape == current.shape:
        current = SEG_MASK_EMA * current + (1.0 - SEG_MASK_EMA) * previous
    return np.clip(current, 0.0, 1.0)


def blur_background(frame, person_mask):
    """Blur the background while retaining a sharp segmented person."""
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (max(1, width // BG_BLUR_DOWNSCALE),
                               max(1, height // BG_BLUR_DOWNSCALE)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), BG_BLUR_SIGMA)
    blurred = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(person_mask, 0.0, 1.0)[..., None]
    result = frame.astype(np.float32) * mask + blurred.astype(np.float32) * (1.0 - mask)
    frame[:] = np.clip(result, 0, 255).astype(np.uint8)


class LandmarkSmoother:
    """Apply EMA smoothing to ordered pixel-space face landmarks."""

    def __init__(self, alpha=LANDMARK_EMA_ALPHA, snap_ratio=LANDMARK_SNAP_RATIO):
        self.alpha = alpha
        self.snap_ratio = snap_ratio
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, faces):
        if self.previous is None or len(faces) != len(self.previous):
            self.previous = [face.copy() for face in faces]
            return [face.copy() for face in faces]
        output = []
        for current, old in zip(faces, self.previous):
            width = max(float(np.linalg.norm(current[FACE_RIGHT_EDGE] -
                                             current[FACE_LEFT_EDGE])), 1.0)
            jump = float(np.linalg.norm(current - old, axis=1).mean())
            if jump > self.snap_ratio * width:
                value = current.copy()
            else:
                value = self.alpha * current + (1.0 - self.alpha) * old
            output.append(value.astype(np.float32))
        self.previous = [face.copy() for face in output]
        return output


def hand_size(points):
    """Return wrist-to-middle-MCP distance in pixels."""
    return max(float(np.linalg.norm(points[WRIST] - points[MIDDLE_MCP])), 1e-6)


def pinch_ratio(points):
    return float(np.linalg.norm(points[THUMB_TIP] - points[8])) / hand_size(points)


def index_reach(points):
    return float(np.linalg.norm(points[8] - points[WRIST])) / hand_size(points)


def fingers_extended(points):
    """Return rotation- and handedness-independent extension flags."""
    size = hand_size(points)
    thumb = (float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_MCP])) / size >
             THUMB_OUT_RATIO and
             np.linalg.norm(points[THUMB_TIP] - points[PINKY_MCP]) >
             np.linalg.norm(points[THUMB_IP] - points[PINKY_MCP]))
    fingers = [thumb]
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(np.linalg.norm(points[tip] - points[WRIST]) >
                       FINGER_EXTENDED_RATIO *
                       np.linalg.norm(points[pip] - points[WRIST]))
    return fingers


def is_open_palm(points):
    return bool(all(fingers_extended(points)))


def largest_hand(hands):
    return max(hands, key=hand_size) if hands else None


class GestureRecognizer:
    """Pure state machine translating synthetic hand landmarks into actions."""

    NEXT = "next"
    TOGGLE = "toggle"

    def __init__(self, pinch_close=PINCH_CLOSE_RATIO, pinch_open=PINCH_OPEN_RATIO,
                 min_index_reach=PINCH_MIN_INDEX_REACH,
                 cooldown_s=PINCH_COOLDOWN_S, palm_hold_s=PALM_HOLD_S):
        self.pinch_close = pinch_close
        self.pinch_open = pinch_open
        self.min_index_reach = min_index_reach
        self.cooldown_s = cooldown_s
        self.palm_hold_s = palm_hold_s
        self.reset()

    def reset(self):
        """Clear all state, including the pinch cooldown timer."""
        self.last_pinch_time = -math.inf
        self._reset_transient()

    def _reset_transient(self):
        """Drop per-frame arming/palm state but keep the cooldown timer."""
        self.pinched = False
        self.armed = False
        self.palm_start = None
        self.palm_latched = False
        self.palm_progress = 0.0

    def update(self, hand_points, now):
        if hand_points is None:
            # Losing the hand disarms and clears palm progress, but the pinch
            # cooldown must survive so re-acquiring the hand cannot double-fire.
            self._reset_transient()
            return None
        event = None
        ratio = pinch_ratio(hand_points)
        if ratio > self.pinch_open:
            # An observed open hand arms the next pinch. A hand first seen
            # already pinched is never armed, so it cannot trigger.
            self.armed = True
        if (not self.pinched and ratio < self.pinch_close and
                index_reach(hand_points) > self.min_index_reach):
            self.pinched = True
            if self.armed and now - self.last_pinch_time >= self.cooldown_s:
                self.armed = False
                self.last_pinch_time = now
                event = self.NEXT
        elif self.pinched and ratio > self.pinch_open:
            self.pinched = False

        open_palm = not self.pinched and is_open_palm(hand_points)
        if open_palm:
            if self.palm_latched:
                self.palm_progress = 0.0
            else:
                if self.palm_start is None:
                    self.palm_start = now
                elapsed = max(0.0, now - self.palm_start)
                self.palm_progress = min(1.0, elapsed / self.palm_hold_s)
                if self.palm_progress >= 1.0:
                    self.palm_latched = True
                    self.palm_progress = 0.0
                    if event is None:
                        event = self.TOGGLE
        else:
            self.palm_start = None
            self.palm_latched = False
            self.palm_progress = 0.0
        return event


class ParticleSystem:
    """Small additive sparkle particle system."""

    PALETTE = np.array(((60, 220, 255), (255, 230, 80), (255, 80, 240),
                        (255, 255, 255)), np.uint8)

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.empty((0, 2), np.float32)
        self.vel = np.empty((0, 2), np.float32)
        self.age = np.empty(0, np.float32)
        self.life = np.empty(0, np.float32)
        self.color = np.empty((0, 3), np.uint8)
        self.size = np.empty(0, np.float32)

    def emit(self, point, count=PARTICLES_PER_TIP):
        angles = self.rng.uniform(0, 2 * math.pi, count)
        speeds = self.rng.uniform(20, 120, count)
        velocity = np.column_stack((np.cos(angles) * speeds,
                                    np.sin(angles) * speeds - 30)).astype(np.float32)
        positions = np.repeat(np.asarray(point, np.float32)[None, :], count, axis=0)
        self.pos = np.concatenate((self.pos, positions))
        self.vel = np.concatenate((self.vel, velocity))
        self.age = np.concatenate((self.age, np.zeros(count, np.float32)))
        self.life = np.concatenate((self.life, self.rng.uniform(
            PARTICLE_LIFE_S[0], PARTICLE_LIFE_S[1], count).astype(np.float32)))
        self.color = np.concatenate((self.color, self.PALETTE[
            self.rng.integers(0, len(self.PALETTE), count)]))
        self.size = np.concatenate((self.size,
                                    self.rng.uniform(2, 6, count).astype(np.float32)))
        if len(self.age) > MAX_PARTICLES:
            keep = slice(len(self.age) - MAX_PARTICLES, None)
            self.pos, self.vel = self.pos[keep], self.vel[keep]
            self.age, self.life = self.age[keep], self.life[keep]
            self.color, self.size = self.color[keep], self.size[keep]

    def update(self, dt):
        if not len(self.age):
            return
        self.age += dt
        self.pos += self.vel * dt
        self.vel *= 0.96
        self.vel[:, 1] -= 8.0 * dt
        keep = self.age < self.life
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.age, self.life = self.age[keep], self.life[keep]
        self.color, self.size = self.color[keep], self.size[keep]

    def draw(self, frame):
        glow = np.zeros_like(frame)
        for position, age, life, color, size in zip(
                self.pos, self.age, self.life, self.color, self.size):
            x, y = np.rint(position).astype(int)
            radius = max(1, int(size * (1.0 - age / life)))
            faded = tuple(int(channel * (1.0 - age / life)) for channel in color)
            cv2.line(glow, (x - 2 * radius, y), (x + 2 * radius, y), faded, 1)
            cv2.line(glow, (x, y - 2 * radius), (x, y + 2 * radius), faded, 1)
            cv2.circle(glow, (x, y), radius, faded, -1)
        cv2.add(frame, glow, dst=frame)
        bloom = cv2.GaussianBlur(glow, (0, 0), 3)
        cv2.add(frame, bloom, dst=frame)

    def clear(self):
        self.__init__()


def key_to_action(key):
    """Map a cv2.waitKeyEx code to an action tuple."""
    if key == -1:
        return None
    if key in KEY_RIGHT:
        return ("next",)
    if key in KEY_LEFT:
        return ("prev",)
    low = key & 0xFF
    if ord("0") <= low <= ord("6"):
        return ("select", low - ord("0"))
    if low in (ord("n"), ord("N")):
        return ("next",)
    if low in (ord("p"), ord("P")):
        return ("prev",)
    if low in (ord("q"), ord("Q"), KEY_ESC):
        return ("quit",)
    return None


def apply_action(state, action, now):
    """Apply a selection, cycle, or toggle action to application state."""
    old = state.filter_idx
    kind = action[0]
    if kind == "select" and 0 <= action[1] < len(FILTER_NAMES):
        state.filter_idx = action[1]
    elif kind == "next":
        state.filter_idx = (state.filter_idx + 1) % len(FILTER_NAMES)
    elif kind == "prev":
        state.filter_idx = (state.filter_idx - 1) % len(FILTER_NAMES)
    elif kind == "toggle":
        if state.filter_idx:
            state.last_active_filter = state.filter_idx
            state.filter_idx = 0
        else:
            state.filter_idx = state.last_active_filter or DEFAULT_FILTER
    if state.filter_idx != old:
        if state.filter_idx:
            state.last_active_filter = state.filter_idx
        state.flash_text = FILTER_NAMES[state.filter_idx]
        state.flash_until = now + FLASH_S


def _landmarks_to_pixels(landmark_sets, width, height):
    output = []
    for landmark_set in landmark_sets or ():
        output.append(np.asarray([(point.x * width, point.y * height)
                                  for point in landmark_set.landmark], np.float32))
    return output


def hands_to_pixels(results, width, height):
    return _landmarks_to_pixels(getattr(results, "multi_hand_landmarks", None),
                                width, height)


def faces_to_pixels(results, width, height):
    return _landmarks_to_pixels(getattr(results, "multi_face_landmarks", None),
                                width, height)


def draw_gesture_feedback(frame, hand, recognizer):
    """Draw pinch state and open-palm hold progress."""
    if hand is None:
        return
    ratio = pinch_ratio(hand)
    if ratio < 0.6:
        color = (60, 255, 60) if recognizer.pinched else (170, 170, 170)
        cv2.line(frame, tuple(np.rint(hand[THUMB_TIP]).astype(int)),
                 tuple(np.rint(hand[8]).astype(int)), color, 2, cv2.LINE_AA)
    if recognizer.palm_progress > 0:
        center = tuple(np.rint(hand[list(PALM_POINTS)].mean(axis=0)).astype(int))
        radius = max(12, int(0.6 * hand_size(hand)))
        cv2.ellipse(frame, center, (radius, radius), -90, 0,
                    360 * recognizer.palm_progress, (80, 230, 255), 5,
                    cv2.LINE_AA)


def draw_hud(frame, state, now):
    """Draw filter name, FPS, controls, and filter-change feedback."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (365, 91), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Filter {state.filter_idx}/6: {FILTER_NAMES[state.filter_idx]}",
                (25, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(state.fps)}", (25, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 240, 140), 2, cv2.LINE_AA)
    hint = "0-6 select | N/P or arrows: next/prev | pinch: next | open palm 1s: on/off | Q/Esc: quit"
    scale = min(0.53, (frame.shape[1] - 20) / max(cv2.getTextSize(
        hint, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 1)[0][0], 1) * 0.53)
    cv2.putText(frame, hint, (10, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)
    if state.flash_text and now < state.flash_until:
        fade = max(0.0, (state.flash_until - now) / FLASH_S)
        text = state.flash_text
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.35, 3)[0]
        position = ((frame.shape[1] - size[0]) // 2,
                    max(60, (frame.shape[0] - size[1]) // 2))
        flash = frame.copy()
        cv2.putText(flash, text, position, cv2.FONT_HERSHEY_DUPLEX, 1.35,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.addWeighted(flash, fade, frame, 1.0 - fade, 0, frame)


@dataclasses.dataclass
class AppState:
    assets: dict[str, Asset]
    filter_idx: int = DEFAULT_FILTER
    last_active_filter: int = DEFAULT_FILTER
    gesture: GestureRecognizer = dataclasses.field(default_factory=GestureRecognizer)
    smoother: LandmarkSmoother = dataclasses.field(default_factory=LandmarkSmoother)
    particles: ParticleSystem = dataclasses.field(default_factory=ParticleSystem)
    seg_mask: typing.Optional[np.ndarray] = None
    flash_text: str = ""
    flash_until: float = 0.0
    prev_time: typing.Optional[float] = None
    fps: float = 0.0
    face_count: int = 0
    hand_count: int = 0


def create_state(seed=None):
    """Create application state and pre-render static graphics."""
    return AppState(assets=build_assets(), particles=ParticleSystem(seed))


@contextlib.contextmanager
def open_models():
    """Create and reliably close all legacy MediaPipe solution models."""
    with contextlib.ExitStack() as stack:
        face_mesh = stack.enter_context(mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES, refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE))
        hands = stack.enter_context(mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE))
        segmenter = stack.enter_context(mp_selfie.SelfieSegmentation(
            model_selection=SEGMENTATION_MODEL))
        yield Models(face_mesh, hands, segmenter)


def process_frame(frame_bgr, state, models, now=None):
    """Process one mirrored BGR frame independently of the capture loop."""
    now = time.perf_counter() if now is None else now
    if state.prev_time is None:
        dt = 0.0
    else:
        dt = min(0.1, max(0.0, now - state.prev_time))
        instantaneous = 1.0 / max(now - state.prev_time, 1e-9)
        state.fps = (FPS_EMA * state.fps + (1.0 - FPS_EMA) * instantaneous
                     if state.fps else instantaneous)
    state.prev_time = now

    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands = hands_to_pixels(models.hands.process(rgb), width, height)
    state.hand_count = len(hands)
    active_hand = largest_hand(hands)
    event = state.gesture.update(active_hand, now)
    if event:
        apply_action(state, (event,), now)

    if state.filter_idx in FACE_FILTERS:
        faces = faces_to_pixels(models.face_mesh.process(rgb), width, height)
        faces = state.smoother.update(faces)
        state.face_count = len(faces)
        for face in faces:
            if state.filter_idx == 1:
                draw_sunglasses(frame_bgr, face, state.assets)
            elif state.filter_idx == 2:
                draw_dog(frame_bgr, face, state.assets)
            elif state.filter_idx == 3:
                draw_makeup(frame_bgr, face)
            elif state.filter_idx == 4:
                apply_big_eyes(frame_bgr, face)
    else:
        state.smoother.reset()
        state.face_count = 0

    if state.filter_idx == FILTER_BG_BLUR:
        result = models.segmenter.process(rgb)
        state.seg_mask = smooth_mask(state.seg_mask, result.segmentation_mask)
        blur_background(frame_bgr, state.seg_mask)
    else:
        state.seg_mask = None

    if state.filter_idx == FILTER_MAGIC:
        for hand in hands:
            for tip in ALL_TIPS:
                state.particles.emit(hand[tip])
        state.particles.update(dt)
        state.particles.draw(frame_bgr)
    else:
        state.particles.clear()

    draw_gesture_feedback(frame_bgr, active_hand, state.gesture)
    draw_hud(frame_bgr, state, now)
    return frame_bgr


def main():
    """Run the mirrored webcam application."""
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    state = create_state()
    try:
        with open_models() as models:
            while True:
                success, frame = capture.read()
                if not success:
                    print("Failed to read a frame from the camera; exiting.")
                    break
                # Gesture geometry is handedness-independent, so mirrored input
                # needs no left/right handedness-label correction.
                frame = cv2.flip(frame, 1)
                process_frame(frame, state, models)
                cv2.imshow(WINDOW_NAME, frame)
                action = key_to_action(cv2.waitKeyEx(1))
                if action == ("quit",):
                    break
                if action:
                    apply_action(state, action, time.perf_counter())
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
