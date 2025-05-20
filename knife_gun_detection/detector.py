import cv2
import torch
from ultralytics import YOLO

# -------------------------------------------------------------------
# 0) DEVICE AYARI: MPS (GPU) varsa ona, yoksa CPU’ya düş
# -------------------------------------------------------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
print(torch.backends.mps.is_available())

'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0) if device == 'cuda' else 'CUDA not available')
'''

# -------------------------------------------------------------------
# GERÇEK ZAMANLI KAMERA TESPİTİ
# -------------------------------------------------------------------
def camera_inference(weights, conf_thres=0.25):
    model = YOLO(weights)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kameraya erişilemiyor.")
        return

    print("Gerçek zamanlı tespit başladı. Çıkmak için 'q' tuşuna basın.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame alınamadı, çıkılıyor.")
            break

        # Inference sırasında da GPU kullanılıyor
        results = model(frame, conf=conf_thres, device=device)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow('Knife-Gun Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    best_weights = '/Users/erinc/GitHub/SecurityOfficer/knife_gun_detection/runs/weapon-det/weights/best.pt'
    camera_inference(best_weights)

# python inputDetector.py -w /Users/erinc/GitHub/SecurityOfficer/knife_gun_detection/runs/weapon-det/weights/best.pt -i eli-silahli-adam-serbest-2241125_amp.jpg