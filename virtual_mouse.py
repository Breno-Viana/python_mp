import math
import time

import mediapipe as mp
import pyautogui as pg
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker

# --- Model and video source setup ---
HANDS_MODEL_PATH = 'landmarks/hand_landmarker.task'
video = cv2.VideoCapture(0)  # opens the default webcam (device index 0)
screen_w, screen_h = pg.size()  # actual monitor resolution, used to map hand position to cursor position

pg.PAUSE = 0  # removes pyautogui's default 0.1s delay after each action, so mouse control feels responsive

# --- MediaPipe HandLandmarker configuration ---
hands_base_options = BaseOptions(model_asset_path=HANDS_MODEL_PATH, delegate=python.BaseOptions.Delegate.GPU)
hands_options = vision.HandLandmarkerOptions(
    base_options=hands_base_options,
    min_hand_detection_confidence=0.5,
    num_hands=2,  # detects up to 2 hands simultaneously
    running_mode=vision.RunningMode.VIDEO,  # synchronous per-frame detection, suited for a simple while loop
)

# --- Hand skeleton connections (pairs of landmark indices to draw lines between) ---
# Defines which of the 21 hand landmarks are anatomically connected, used to draw the "hand skeleton" overlay
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9),
               (9, 10), (10, 11), (11, 12),
               (9, 13),
               (13, 14), (14, 15), (15, 16),
               (13, 17),
               (0, 17), (17, 18), (18, 19), (19, 20), ]

# --- Tunable constants ---
SENSITIVITY_MARGIN = 0.35 # fraction of the webcam frame's edges to ignore when mapping hand -> screen position
SMOOTHING = 0.25           # cursor smoothing factor (0-1): lower = smoother/laggier, higher = more responsive/jittery
CLICK_THRESHOLD = 30       # max pixel distance (webcam space) between fingertips to count as a "pinch"/click
MIN_RANGE = SENSITIVITY_MARGIN       # lower bound of the "active zone" (0-1 scale) used by remap() for cursor control
MAX_RANGE = 1 - SENSITIVITY_MARGIN   # upper bound of the "active zone" (0-1 scale)

# Landmark indices for the 4 non-thumb fingertips, and their corresponding base knuckles (MCP joints).
# Comparing tip vs. MCP vertical position is how the script determines if a finger is extended or curled.
TIPS_LANDMARKS = [8, 12, 16, 20]  # index, middle, ring, pinky fingertip indices
MCP_LANDMARKS = [5, 9, 13, 17]    # index, middle, ring, pinky base-knuckle (MCP) indices

start = time.time()  # reference start time, used to compute strictly increasing per-frame timestamps

# --- Per-hand gesture state (persists across frames) ---
hands_closed = 0                    # counts how many hands are currently in a "closed fist" state (used as an exit condition)
was_touching_tips = [False, False]  # debounce state for the right-physical-hand clicks: [index-thumb pinch, middle-thumb pinch]
hand_was_closed = [False, False]    # debounce state per hand slot (idx 0/1): was this hand closed in the previous frame?
is_dragging = False                 # tracks whether a mouseDown drag is currently active (left-physical-hand gesture)
prev_mouse_x, prev_mouse_y = screen_w // 2, screen_h // 2  # smoothed cursor position, starts at screen center


def distance(p1, p2):
    """Euclidean distance between two (x, y) points, using the Pythagorean theorem."""
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


def normalize(point_lm, width, height):
    """Converts a single normalized (0-1) landmark coordinate into a pixel coordinate,
    given a target width/height (either the webcam frame or the screen)."""
    x, y = point_lm
    return round(x * width), round(y * height)


def higher(point_1: int, point_2: int):
    """Returns True if point_1 is greater than point_2.
    Used to compare Y coordinates: in image space, a SMALLER y means higher up on screen,
    so 'greater y' here effectively checks if point_1 sits LOWER than point_2 (see usage below)."""
    return point_1 > point_2


def is_tip_higher_than_mcp(points_tip: list[tuple[int, int]], points_mcp: list[tuple[int, int]]):
    """
    For each (fingertip, base-knuckle) pair, checks whether the fingertip's Y coordinate
    is greater than the knuckle's Y coordinate.
    Since Y grows downward in image coordinates, tip_y > mcp_y means the tip is BELOW
    the knuckle, i.e. the finger is curled/folded rather than extended.
    Returns a list of booleans, one per finger, in the same order as the input lists.
    """
    return [
        higher(tip[1], mcp[1])
        for tip, mcp in zip(points_tip, points_mcp)
    ]


def normalize_all(landmarks: list):
    """
    Converts a list of MediaPipe landmark objects (each with .x/.y in 0-1 range)
    into a list of pixel coordinates, scaled to the SCREEN resolution (screen_w, screen_h).
    Returns a plain list (not a generator), so the result can be indexed/sliced/reused.
    """
    return [
        normalize((lmk.x, lmk.y), screen_w, screen_h)
        for lmk in landmarks
    ]


def remap(value, in_min, in_max, out_max):
    """
    Maps `value` from the input range [in_min, in_max] to an output scale of out_max.
    The value is clamped to [in_min, in_max] first, so a fingertip that strays outside
    the "active zone" still produces a valid, in-bounds screen coordinate instead of
    overshooting past the edges of the display.
    NOTE: out_min is accepted as a parameter but not actually used in the formula below
    (the multiplication only scales by out_max) -- this only produces correct results
    when out_min is 0, which is the case everywhere this function is currently called.
    """
    value = max(in_min, min(in_max, value))
    return (value - in_min) / (in_max - in_min) * out_max


def scroll(hand: str | None):
    """
    Performs a single scroll tick in a fixed direction depending on which hand triggered it.
    'left' scrolls down (-1), 'right' scrolls up (+1). Does nothing if the hand is None.
    NOTE: this scrolls a fixed amount per call, not proportional to any gesture intensity --
    the calling code is expected to only call this once per frame while the gesture is held,
    which (at ~30fps) produces a continuous, fixed-speed scroll for as long as the gesture is active.
    """
    if hand is None:
        return
    if hand == 'left':
        pg.scroll(-1)
    if hand == 'right':
        pg.scroll(1)


with HandLandmarker.create_from_options(options=hands_options) as hands_landmark:
    while video.isOpened():
        # --- Frame capture ---
        ok, frame = video.read()
        if not ok:
            break  # stop if the webcam feed fails or ends

        h, w, _ = frame.shape  # webcam frame dimensions, used to convert normalized landmarks to pixel space
        label = "frame " + str(w) + "x" + str(h)
        cv2.flip(frame, 1, frame)  # mirror the frame horizontally, so hand movement feels natural (like a mirror)

        # --- Prepare frame for MediaPipe (expects RGB, OpenCV captures in BGR) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- Run hand detection for this frame ---
        # Timestamp must be strictly increasing across calls; real elapsed time keeps internal tracking accurate.
        timestamp_ms = int((time.time() - start) * 1000)
        result = hands_landmark.detect_for_video(image=hand_image, timestamp_ms=timestamp_ms)

        if result.hand_landmarks:
            # Iterate over each detected hand (up to 2, since num_hands=2)
            for idx, hand_landmarks in enumerate(result.hand_landmarks):

                # --- Convert all 21 landmarks to webcam-frame pixel coordinates, for drawing only ---
                points = []
                for lm in hand_landmarks:
                    cx, cy = round(lm.x * w), round(lm.y * h)
                    points.append((cx, cy))

                # Draw the hand "skeleton" lines between connected landmarks
                for start_idx, end_idx in CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

                # Draw each landmark as a labeled dot (index number + circle), useful for debugging
                for ix, point in enumerate(points):
                    cv2.putText(frame, str(ix), point, cv2.FONT_HERSHEY_PLAIN, 1, (240, 0, 40), 1)
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)

                if result.handedness:
                    # Get the Left/Right classification matching this specific hand (same index as hand_landmarks)
                    hand = result.handedness[idx]
                    hand_label = hand[0].display_name

                    """
                    The hand_label indicates that this is the left hand, but it is actually the right;
                    for the section marked 'Left', please bear in mind that we are dealing with the right hand.
                    """
                    # ============================================================
                    # LABELED "Left" (physically the RIGHT hand): cursor movement + drag + scroll-up
                    # ============================================================
                    if hand_label == 'Left':
                        index_tip = hand_landmarks[TIPS_LANDMARKS[0]]
                        thumb_tip = hand_landmarks[4]

                        # --- Cursor movement: map index fingertip position (active zone) to full screen ---
                        target_x = remap(index_tip.x, MIN_RANGE, MAX_RANGE, 0, screen_w)
                        target_y = remap(index_tip.y, MIN_RANGE, MAX_RANGE, 0, screen_h)

                        # Smooth the cursor movement (linear interpolation) to reduce frame-to-frame jitter
                        smooth_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING
                        smooth_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING
                        pg.moveTo(x=round(smooth_x), y=round(smooth_y))
                        prev_mouse_x, prev_mouse_y = smooth_x, smooth_y  # remember position for next frame's smoothing

                        # --- Gather remaining fingertip and MCP (knuckle) landmarks for finger-state detection ---
                        middle_tip = hand_landmarks[TIPS_LANDMARKS[1]]
                        ring_tip = hand_landmarks[TIPS_LANDMARKS[2]]
                        pinky_tip = hand_landmarks[TIPS_LANDMARKS[3]]

                        # Fingertip positions, normalized to SCREEN scale (used only for the higher()/closed-fist check)
                        tips_points_normalized = normalize_all([index_tip, middle_tip, ring_tip, pinky_tip])

                        index_mcp = hand_landmarks[MCP_LANDMARKS[0]]
                        middle_mcp = hand_landmarks[MCP_LANDMARKS[1]]
                        ring_mcp = hand_landmarks[MCP_LANDMARKS[2]]
                        pinky_mcp = hand_landmarks[MCP_LANDMARKS[3]]

                        # Base-knuckle positions, also normalized to SCREEN scale, for the same comparison
                        mcp_points_normalized = normalize_all([index_mcp, middle_mcp, ring_mcp, pinky_mcp])

                        # --- Drag gesture: thumb + middle finger pinch toggles mouseDown/mouseUp ---
                        thumb_normalized = normalize((thumb_tip.x, thumb_tip.y), w, h)
                        middle_normalized = normalize((middle_tip.x, middle_tip.y), w, h)

                        dist_thumb_middle = distance(thumb_normalized, middle_normalized)
                        is_touching_middle = dist_thumb_middle < CLICK_THRESHOLD

                        # On the transition into a pinch, press and HOLD the left mouse button (start dragging)
                        if is_touching_middle and not is_dragging:
                            pg.mouseDown(prev_mouse_x, prev_mouse_y)
                        # On the transition out of a pinch, release the button (stop dragging)
                        if not is_touching_middle and is_dragging:
                            pg.mouseUp(prev_mouse_x, prev_mouse_y)
                        is_dragging = is_touching_middle

                        # --- Closed-fist detection: True per finger if that fingertip is curled (below its MCP) ---
                        is_higher = is_tip_higher_than_mcp(tips_points_normalized, mcp_points_normalized)

                        # Fist is considered "closed" only if ALL four fingers are curled
                        right_is_closed = all(is_higher)

                        # Debounced closed-fist counter: increments/decrements hands_closed only on state transitions
                        # (see the earlier bug discussion: the "before" state must be captured BEFORE overwriting it)
                        was_closed_before = hand_was_closed[idx]
                        if right_is_closed and not was_closed_before:
                            hands_closed += 1
                        if not right_is_closed and was_closed_before:
                            hands_closed -= 1
                        hand_was_closed[idx] = right_is_closed

                        # --- Scroll trigger: ring + pinky fingers curled (last two entries of is_higher) ---
                        if all(is_higher[2:]):
                            scroll('right')  # scrolls up (+1), see scroll() docstring

                    """
                    The hand_label indicates that this is the right hand, but it is actually the left;
                    for the section marked 'Right', please bear in mind that we are dealing with the left hand.
                    """
                    # ============================================================
                    # LABELED "Right" (physically the LEFT hand): left-click, right-click + scroll-down
                    # ============================================================
                    if hand_label == 'Right':
                        thumb_tip = hand_landmarks[4]
                        index_tip = hand_landmarks[TIPS_LANDMARKS[0]]

                        thumb_point = thumb_tip.x, thumb_tip.y
                        index_point = index_tip.x, index_tip.y

                        # Normalized using WEBCAM frame dimensions (w, h), not screen dimensions --
                        # these are only used to measure pinch distance within the webcam image itself
                        thumb_normalized = normalize(thumb_point, w, h)
                        index_normalized = normalize(index_point, w, h)

                        # --- Left click: thumb + index finger pinch ---
                        dist_thumb_index = distance(thumb_normalized, index_normalized)
                        is_touching_index = dist_thumb_index < CLICK_THRESHOLD

                        # Fires once, only on the "not touching -> touching" transition (debounced click)
                        if is_touching_index and not was_touching_tips[0]:
                            pg.click(x=round(prev_mouse_x), y=round(prev_mouse_y))
                        was_touching_tips[0] = is_touching_index

                        # --- Right-click: thumb + middle finger pinch ---
                        middle_tip = hand_landmarks[12]
                        middle_point = middle_tip.x, middle_tip.y
                        middle_normalized = normalize(middle_point, w, h)

                        dist_thumb_middle = distance(thumb_normalized, middle_normalized)
                        # print(dist_thumb_middle)
                        is_touching_middle = dist_thumb_middle < CLICK_THRESHOLD

                        # Tracked independently from the left-click gesture (separate debounce slot)
                        if is_touching_middle and not was_touching_tips[1]:
                            pg.rightClick(x=round(prev_mouse_x), y=round(prev_mouse_y))
                        was_touching_tips[1] = is_touching_middle

                        # --- Gather fingertip/MCP landmarks for this hand's own closed-fist + scroll detection ---
                        middle_tip = hand_landmarks[TIPS_LANDMARKS[1]]
                        ring_tip = hand_landmarks[TIPS_LANDMARKS[2]]
                        pinky_tip = hand_landmarks[TIPS_LANDMARKS[3]]

                        tips_points_normalized = normalize_all([index_tip, middle_tip, ring_tip, pinky_tip])

                        index_mcp = hand_landmarks[MCP_LANDMARKS[0]]
                        middle_mcp = hand_landmarks[MCP_LANDMARKS[1]]
                        ring_mcp = hand_landmarks[MCP_LANDMARKS[2]]
                        pinky_mcp = hand_landmarks[MCP_LANDMARKS[3]]

                        mcp_points_normalized = normalize_all([index_mcp, middle_mcp, ring_mcp, pinky_mcp])

                        # Per-finger curled/extended state, same logic as the other hand
                        is_higher = is_tip_higher_than_mcp(tips_points_normalized, mcp_points_normalized)

                        left_is_closed = all(is_higher)

                        # Debounced closed-fist counter, using this hand's own slot (idx) in hand_was_closed
                        was_closed_before = hand_was_closed[idx]
                        if left_is_closed and not was_closed_before:
                            hands_closed += 1
                        if not left_is_closed and was_closed_before:
                            hands_closed -= 1
                        hand_was_closed[idx] = left_is_closed

                        # --- Scroll trigger: ring + pinky fingers curled ---
                        if all(is_higher[2:]):
                            scroll('left')  # scrolls down (-1), see scroll() docstring

        # cv2.imshow(label, frame)  # webcam preview window disabled; mouse control runs "headless"
        # cv2.waitKey(1)            # disabled along with imshow -- no OpenCV window means no keyboard capture here

        # Exit condition: stop the script once BOTH hands are simultaneously detected as closed fists
        if hands_closed == 2:
            break

video.release()  # release the webcam device
cv2.destroyAllWindows()  # close any OpenCV windows (no-op here since imshow is disabled)