import cv2
import mediapipe as mp

# Mediapipe'in el alg,ılama modeli
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Kamerayı başlat
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """
    Parmakların açık mı kapalı mı olduğunu kontrol eder.
    """
    finger_tips = [8, 12, 16, 20]  # İşaret, orta, yüzük, serçe
    thumb_tip = 4  # Başparmak

    fingers = []

    # Başparmağın durumu
    if hand_landmarks[thumb_tip].x < hand_landmarks[thumb_tip - 1].x:
        fingers.append(1)  # Açık
    else:
        fingers.append(0)  # Kapalı

    # Diğer parmakların durumu
    for tip in finger_tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers.append(1)  # Açık
        else:
            fingers.append(0)  # Kapalı

    return fingers.count(1)  # Açık parmak sayısını döndür

def process_frame():
    """
    Kameradan gelen görüntüyü işler ve parmak sayısını döndürür.
    """
    ret, frame = cap.read()
    if not ret:
        return None, None

    # Görüntüyü yansıt ve RGB renk uzayına dönüştür
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe ile el algılama
    result = hands.process(rgb_frame)

    finger_count = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Landmarkları çiz
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Parmak sayısını hesapla
            finger_count = count_fingers(hand_landmarks.landmark)

    return frame, finger_count

def release_resources():
    """
    Kamera ve kaynakları serbest bırakır.
    """
    cap.release()
    cv2.destroyAllWindows()
