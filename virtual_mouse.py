import math
import time
import mediapipe as mp
import pyautogui as pg
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker

HANDS_MODEL_PATH = 'landmarks/hand_landmarker.task'
video = cv2.VideoCapture(0)
screen_w, screen_h = pg.size()



pg.PAUSE = 0

baseOptions = BaseOptions(model_asset_path=HANDS_MODEL_PATH, delegate=python.BaseOptions.Delegate.CPU)
options = vision.HandLandmarkerOptions(
    base_options=baseOptions,
    min_hand_detection_confidence=0.5,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)
start = time.time()

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

SENSITIVITY_MARGIN = 0.3

MIN_RANGE = SENSITIVITY_MARGIN
MAX_RANGE = 1 - SENSITIVITY_MARGIN

was_touching_tip = [False,False,False,False,False]
prev_mouse_x, prev_mouse_y = screen_w//2, screen_h//2
SMOOTHING = 0.5
CLICK_THRESHOLD = 40


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


def normalize(point, width, height):
    x, y = point
    return round(x * width), round(y * height)

def remap(value, in_min, in_max, max_view):
    value = max(in_min, min(in_max, value))
    return (value - in_min) / (in_max - in_min) * max_view






with HandLandmarker.create_from_options(options=options) as hands_landmark:
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        h, w, _ = frame.shape
        label = "frame " + str(w) + "x" + str(h)
        cv2.flip(frame, 1, frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start) * 1000)
        result = hands_landmark.detect_for_video(image=hand_image, timestamp_ms=timestamp_ms)

        if result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                points = []
                for lm in hand_landmarks:
                    cx, cy = round(lm.x * w), round(lm.y * h)
                    points.append((cx, cy))

                for start_idx, end_idx in CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)

                if result.handedness:
                    hand = result.handedness[idx]
                    hand_label = hand[0].display_name


                    if hand_label == 'Left':
                        index_tip = hand_landmarks[8]

                        target_x = remap(index_tip.x, MIN_RANGE, MAX_RANGE,  screen_w)
                        target_y = remap(index_tip.y, MIN_RANGE, MAX_RANGE,  screen_h)

                        smooth_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING
                        smooth_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING
                        pg.moveTo(x=round(smooth_x), y=round(smooth_y))
                        prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

                    if hand_label == 'Right':
                        thumb_tip = hand_landmarks[4]
                        index_finger_tip = hand_landmarks[8]

                        thumb_point = thumb_tip.x, thumb_tip.y
                        index_point = index_finger_tip.x, index_finger_tip.y

                        thumb_normalized = normalize(thumb_point, screen_w, screen_h)
                        index_normalized = normalize(index_point, screen_w, screen_h)

                        dist_index = distance(thumb_normalized, index_normalized)

                        is_touching_index = dist_index < CLICK_THRESHOLD

                        if is_touching_index and not was_touching_tip[0]:
                            pg.click(x=round(prev_mouse_x), y=round(prev_mouse_y))
                        was_touching_tip[0] = is_touching_index


                        middle_finger_tip = hand_landmarks[12]
                        middle_finger_point = middle_finger_tip.x, middle_finger_tip.y
                        middle_normalized = normalize(middle_finger_point, screen_w, screen_h)
                        dist_middle = distance(thumb_normalized, middle_normalized)

                        is_touching_middle = dist_middle < CLICK_THRESHOLD

                        if is_touching_middle and not was_touching_tip[1]:
                            pg.rightClick(x=round(prev_mouse_x), y=round(prev_mouse_y))
                        was_touching_tip[1] = is_touching_middle


        cv2.imshow(label, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()

