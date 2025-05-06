import mediapipe as mp

def init_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return hands, mp.solutions.drawing_utils

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    return [coord for point in lm for coord in (point.x, point.y, point.z)]
