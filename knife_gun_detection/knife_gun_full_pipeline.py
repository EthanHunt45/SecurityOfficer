import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# -------------------------------------------------------------------
# 0) DEVICE AYARI: MPS (GPU) varsa ona, yoksa CPU’ya düş
# -------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0) if device == 'cuda' else 'CUDA not available')

'''
# -------------------------------------------------------------------
# 0) DEVICE AYARI: MPS (GPU) varsa ona, yoksa CPU’ya düş
# -------------------------------------------------------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
print(torch.backends.mps.is_available())
'''

# -------------------------------------------------------------------
# 1) DATA YAML HAZIRLIK
# -------------------------------------------------------------------
def prepare_data_yaml():
    train_images = r'C:\Users\erinc\OneDrive\Desktop\SecurityOfficer\knife_gun_detection\guns-knives_dataset\train\images'
    val_images   = r'C:\Users\erinc\OneDrive\Desktop\SecurityOfficer\knife_gun_detection\guns-knives_dataset\validation\images'
    test_images  = r'C:\Users\erinc\OneDrive\Desktop\SecurityOfficer\knife_gun_detection\guns-knives_dataset\test\images'
    yaml_path    = r'C:\Users\erinc\OneDrive\Desktop\SecurityOfficer\knife_gun_detection\guns-knives_dataset\data.yaml'

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
        epochs=200,
        batch=16,
        imgsz=640,
        device=device,
        project='runs',
        name='weapon-det',
        exist_ok=True,
        patience=20  # Early stopping: 20 epoch boyunca iyileşme yoksa durur
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
# 4) ANA AKIŞ
# -------------------------------------------------------------------
if __name__ == '__main__':
    data_yaml = prepare_data_yaml()
    train_model(data_yaml)
    plot_training_metrics()
    best_weights = os.path.join('runs','train','weapon-det','weights','best.pt')
