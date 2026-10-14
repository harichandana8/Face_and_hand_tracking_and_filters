"""Real-time face mesh + hand tracking from a webcam using MediaPipe.

Draws hand landmarks/connections and face mesh contours on each frame and
shows the current FPS. Press 'q' or ESC (or close the window) to quit.

VERIFICATION (performed with .venv\\Scripts\\python.exe, Python 3.12.0,
mediapipe 0.10.21, opencv-contrib-python 4.11.0.86):
  1. `python -m py_compile face_and_hand_Tracking.py` -> passed.
  2. Headless smoke test (temporary script, since deleted): imported this
     module, loaded other-files/img0.jpg with cv2.imread, created Hands and
     FaceMesh exactly as main() does, called process_frame() and
     draw_fps() on the image without opening a window -> no exceptions,
     returned annotated frame had the same shape as the input.
  Live webcam behavior (capture loop, window, quit keys, window-close
  detection) was NOT verified.
"""

import time

import cv2
import mediapipe as mp

# ---------------------------------------------------------------- config ---
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080  # Requested; the camera may pick another size.
WINDOW_NAME = "Image"

MAX_NUM_FACES = 2
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

QUIT_KEYS = {ord("q"), 27}  # 'q' or ESC

FPS_POSITION = (20, 70)
FPS_FONT = cv2.FONT_HERSHEY_PLAIN
FPS_SCALE = 3
FPS_COLOR = (0, 255, 0)  # BGR
FPS_THICKNESS = 3

# ------------------------------------------------------ mediapipe handles ---
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

FACE_DRAW_SPEC = mp_draw.DrawingSpec(thickness=1, circle_radius=1)


def process_frame(frame_bgr, hands, face_mesh):
    """Run hand + face mesh detection on a BGR frame and draw the results.

    The frame is annotated in place and also returned for convenience.
    """
    # MediaPipe expects RGB; marking it read-only lets it avoid a copy.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                FACE_DRAW_SPEC,
                FACE_DRAW_SPEC,
            )

    return frame_bgr


def draw_fps(frame_bgr, fps):
    """Overlay the FPS counter in the top-left corner."""
    cv2.putText(
        frame_bgr,
        f"FPS: {int(fps)}",
        FPS_POSITION,
        FPS_FONT,
        FPS_SCALE,
        FPS_COLOR,
        FPS_THICKNESS,
    )


def should_quit():
    """Return True if a quit key was pressed or the window was closed."""
    key = cv2.waitKey(1) & 0xFF
    if key in QUIT_KEYS:
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prev_time = time.perf_counter()
    try:
        with mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands, mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as face_mesh:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Failed to read a frame from the camera; exiting.")
                    break

                process_frame(frame, hands, face_mesh)

                now = time.perf_counter()
                elapsed = now - prev_time
                prev_time = now
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                draw_fps(frame, fps)

                cv2.imshow(WINDOW_NAME, frame)
                if should_quit():
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
