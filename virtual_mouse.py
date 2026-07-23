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

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9),
               (9, 10), (10, 11), (11, 12),
               (9, 13),
               (13, 14), (14, 15), (15, 16),
               (13, 17),
               (0, 17), (17, 18), (18, 19), (19, 20), ]
SENSITIVITY_MARGIN = 0.35
SMOOTHING = 0.3
CLICK_THRESHOLD = 30
MIN_RANGE = SENSITIVITY_MARGIN
MAX_RANGE = 1 - SENSITIVITY_MARGIN
TIPS_LANDMARKS = [8, 12, 16, 20]
MCP_LANDMARKS = [5, 9, 13, 17]

start = time.time()
hands_closed = 0
was_touching_tips = [False, False]
hand_was_closed = [False, False]
is_dragging = False
prev_mouse_x, prev_mouse_y = screen_w // 2, screen_h // 2


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


def normalize(point_lm, width, height):
    x, y = point_lm
    return round(x * width), round(y * height)


def higher(point_1: int, point_2: int):
    return point_1 > point_2


def is_tip_higher_than_mcp(points_tip: list[tuple[int, int]], points_mcp: list[tuple[int, int]]):
    return [
        higher(tip[1], mcp[1])
        for tip, mcp in zip(points_tip, points_mcp)
    ]


def normalize_all(landmarks: list):
    return [
        normalize((lmk.x, lmk.y), screen_w, screen_h)
        for lmk in landmarks
    ]


def remap(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


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
                for ix, point in enumerate(points):
                    cv2.putText(frame, str(ix), point, cv2.FONT_HERSHEY_PLAIN, 1, (240, 0, 40), 1)
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)

                if result.handedness:
                    hand = result.handedness[idx]
                    hand_label = hand[0].display_name

                    """
                    The hand_label indicates that this is the left hand, but it is actually the right;
                    for the section marked 'Left', please bear in mind that we are dealing with the right hand.
                    """
                    if hand_label == 'Left':
                        index_tip = hand_landmarks[TIPS_LANDMARKS[0]]
                        thumb_tip = hand_landmarks[4]

                        target_x = remap(index_tip.x, MIN_RANGE, MAX_RANGE, 0, screen_w)
                        target_y = remap(index_tip.y, MIN_RANGE, MAX_RANGE, 0, screen_h)

                        smooth_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING
                        smooth_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING
                        pg.moveTo(x=round(smooth_x), y=round(smooth_y))
                        prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

                        middle_tip = hand_landmarks[TIPS_LANDMARKS[1]]
                        ring_tip = hand_landmarks[TIPS_LANDMARKS[2]]
                        pinky_tip = hand_landmarks[TIPS_LANDMARKS[3]]

                        tips_points_normalized = normalize_all([index_tip, middle_tip, ring_tip, pinky_tip])

                        index_mcp = hand_landmarks[MCP_LANDMARKS[0]]
                        middle_mcp = hand_landmarks[MCP_LANDMARKS[1]]
                        ring_mcp = hand_landmarks[MCP_LANDMARKS[2]]
                        pinky_mcp = hand_landmarks[MCP_LANDMARKS[3]]

                        mcp_points_normalized = normalize_all([index_mcp, middle_mcp, ring_mcp, pinky_mcp])

                        thumb_normalized = normalize((thumb_tip.x, thumb_tip.y), w, h)
                        middle_normalized = normalize((middle_tip.x, middle_tip.y), w, h)

                        dist_thumb_middle = distance(thumb_normalized, middle_normalized)

                        is_touching_middle = dist_thumb_middle < CLICK_THRESHOLD

                        if is_touching_middle and not is_dragging:
                            pg.mouseDown(prev_mouse_x, prev_mouse_y)
                        if not is_touching_middle and is_dragging:
                            pg.mouseUp(prev_mouse_x, prev_mouse_y)
                        is_dragging = is_touching_middle

                        is_higher = is_tip_higher_than_mcp(tips_points_normalized, mcp_points_normalized)

                        right_is_closed = all(is_higher)





                        was_closed_before = hand_was_closed[idx]
                        if right_is_closed and not was_closed_before:
                            hands_closed += 1
                        if not right_is_closed and was_closed_before:
                            hands_closed -= 1
                        hand_was_closed[idx] = right_is_closed


                    """
                    The hand_label indicates that this is the right hand, but it is actually the left;
                    for the section marked 'Right', please bear in mind that we are dealing with the left hand.
                    """
                    if hand_label == 'Right':
                        thumb_tip = hand_landmarks[4]
                        index_tip = hand_landmarks[TIPS_LANDMARKS[0]]

                        thumb_point = thumb_tip.x, thumb_tip.y
                        index_point = index_tip.x, index_tip.y

                        thumb_normalized = normalize(thumb_point, w, h)
                        index_normalized = normalize(index_point, w, h)

                        dist_thumb_index = distance(thumb_normalized, index_normalized)

                        is_touching_index = dist_thumb_index < CLICK_THRESHOLD

                        if is_touching_index and not was_touching_tips[0]:
                            pg.click(x=round(prev_mouse_x), y=round(prev_mouse_y))
                        was_touching_tips[0] = is_touching_index

                        middle_tip = hand_landmarks[12]
                        middle_point = middle_tip.x, middle_tip.y
                        middle_normalized = normalize(middle_point, w, h)

                        dist_thumb_middle = distance(thumb_normalized, middle_normalized)
                        # print(dist_thumb_middle)
                        is_touching_middle = dist_thumb_middle < CLICK_THRESHOLD

                        if is_touching_middle and not was_touching_tips[1]:
                            pg.rightClick(x=round(prev_mouse_x), y=round(prev_mouse_y))
                        was_touching_tips[1] = is_touching_middle

                        middle_tip = hand_landmarks[TIPS_LANDMARKS[1]]
                        ring_tip = hand_landmarks[TIPS_LANDMARKS[2]]
                        pinky_tip = hand_landmarks[TIPS_LANDMARKS[3]]

                        tips_points_normalized = normalize_all([index_tip, middle_tip, ring_tip, pinky_tip])

                        index_mcp = hand_landmarks[MCP_LANDMARKS[0]]
                        middle_mcp = hand_landmarks[MCP_LANDMARKS[1]]
                        ring_mcp = hand_landmarks[MCP_LANDMARKS[2]]
                        pinky_mcp = hand_landmarks[MCP_LANDMARKS[3]]

                        mcp_points_normalized = normalize_all([index_mcp, middle_mcp, ring_mcp, pinky_mcp])

                        is_higher = is_tip_higher_than_mcp(tips_points_normalized, mcp_points_normalized)

                        left_is_closed = all(is_higher)

                        was_closed_before = hand_was_closed[idx]
                        if left_is_closed and not was_closed_before:
                            hands_closed += 1
                        if not left_is_closed and was_closed_before:
                            hands_closed -= 1
                        hand_was_closed[idx] = left_is_closed



        cv2.imshow(label, frame)
        cv2.waitKey(1)
        if hands_closed == 2:
            break

video.release()
cv2.destroyAllWindows()
