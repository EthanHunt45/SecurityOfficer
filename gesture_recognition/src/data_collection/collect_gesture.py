import cv2
import mediapipe as mp
import csv
import os
import argparse

# MediaPipe setup
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_draw

# Landmarkları vektöre çevirir
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    handLms = results.multi_hand_landmarks[0]
    data = []
    for lm in handLms.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data  # 21*3 = 63 boyutlu vektör

# Veri toplama fonksiyonu
def collect_data(gesture, samples, seq_len, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{gesture}.csv")
    hands, mp_draw = init_mediapipe()
    cap = cv2.VideoCapture(0)

    collected = 0
    sequence = []
    print(f"Collecting '{gesture}' samples: {samples} sequences of length {seq_len}")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        while collected < samples:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            data = extract_landmarks(results)

            if data:
                sequence.append(data)
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

                if len(sequence) == seq_len:
                    writer.writerow([gesture] + sum(sequence, []))
                    collected += 1
                    print(f"Collected {collected}/{samples}")
                    sequence = []

            cv2.imshow('Collector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done data collection.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gesture', required=True)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--output_dir', default='./data')
    args = parser.parse_args()
    collect_data(args.gesture, args.samples, args.seq_len, args.output_dir)