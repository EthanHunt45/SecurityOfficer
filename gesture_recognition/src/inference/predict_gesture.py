import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

SEQ_LEN = 30
FEATURE_DIM = 63
GESTURE_CLASSES = ['Rigt_Hand_Threat_Sign']  # Eğitici sırasında kaydedilen sınıflar burada listelenecek

# MediaPipe setup
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7)
    return hands, mp.solutions.drawing_utils

# Landmark extract
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    data = []
    for lm in results.multi_hand_landmarks[0].landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data

if __name__ == '__main__':
    model = load_model('gesture_lstm.h5')
    # TODO: GESTURE_CLASSES = ['Help', 'Threat', 'Safe', ...]

    hands, mp_draw = init_mediapipe()
    cap = cv2.VideoCapture(0)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        data = extract_landmarks(results)

        if data:
            sequence.append(data)
            if len(sequence) > SEQ_LEN:
                sequence.pop(0)
            if len(sequence) == SEQ_LEN:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                idx = np.argmax(res)
                gesture = GESTURE_CLASSES[idx]
                print(f"Detected gesture: {gesture} ({res[idx]:.2f})")

        cv2.imshow('Predictor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()