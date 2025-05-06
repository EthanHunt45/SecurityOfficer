import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# -------------------------------------------------------------------
# 0) DEVICE AYARI: MPS (GPU) varsa ona, yoksa CPU’ya düş
# -------------------------------------------------------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
print(torch.backends.mps.is_available())

# -------------------------------------------------------------------
# 1) DATA YAML HAZIRLIK
# -------------------------------------------------------------------
def prepare_data_yaml():
    train_images = '/Users/erinc/GitHub/SecurityOfficer/knife_gun_detection/guns-knives_dataset/train/images'
    val_images   = '/Users/erinc/GitHub/SecurityOfficer/knife_gun_detection/guns-knives_dataset/validation/images'
    test_images  = '/Users/erinc/GitHub/SecurityOfficer/knife_gun_detection/guns-knives_dataset/test/images'
    yaml_path    = '/Users/erinc/GitHub/SecurityOfficer/knife_gun_detection/guns-knives_dataset/data.yaml'

    with open(yaml_path, 'w') as f:
        f.write(f"train: {train_images}\n")
        f.write(f"val:   {val_images}\n")
        f.write(f"test:  {test_images}\n")
        f.write("nc: 2\n")
        f.write("names: ['knife','gun']\n")

    print(f"Created data config at {yaml_path}")
    return yaml_path

# -------------------------------------------------------------------
# 2) EĞİTİM (MPS GPU veya CPU)
# -------------------------------------------------------------------
def train_model(data_yaml):
    # Model, ağırlıkları ve cihaza taşıma
    model = YOLO('yolov5s.pt')
    results = model.train(
        data=data_yaml,
        epochs=1,
        batch=16,
        imgsz=640,
        device=device,       # artık mps veya cpu
        project='runs',
        name='weapon-det',
        exist_ok=True
    )
    print("Training finished, best weights saved to:", results.best_checkpoint)
    return results

# -------------------------------------------------------------------
# 3) GRAFİKSEL METRİK ÇİZİMİ
# -------------------------------------------------------------------
def plot_training_metrics():
    csv_path = os.path.join('runs', 'train', 'weapon-det', 'results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for col in ['box_loss','cls_loss','obj_loss','mAP50']:
            if col in df.columns:
                plt.figure()
                plt.plot(df['epoch'], df[col])
                plt.title(col)
                plt.xlabel('epoch')
                plt.ylabel(col)
                plt.grid(True)
                plt.show()
    else:
        img_path = os.path.join('runs', 'train', 'weapon-det', 'results.png')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10,6))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            print('Ne results.csv ne de results.png bulundu. Eğitim çıktısını kontrol edin.')

# -------------------------------------------------------------------
# 4) GERÇEK ZAMANLI KAMERA TESPİTİ
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

# -------------------------------------------------------------------
# 5) ANA AKIŞ
# -------------------------------------------------------------------
if __name__ == '__main__':
    data_yaml = prepare_data_yaml()
    train_model(data_yaml)
    plot_training_metrics()
    best_weights = os.path.join('runs','train','weapon-det','weights','best.pt')
