import os
import cv2
import torch
from ultralytics import YOLO
import argparse

def get_device():
    # Öncelikle MPS (Apple GPU), yoksa CUDA, yoksa CPU
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def process_image(model, device, img_path, out_path, conf_thres):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Resim açılamadı: {img_path}")
        return

    # Inference
    results = model(img, conf=conf_thres, device=device)[0]
    # Bounding box çizimi
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    # Çıktıyı kaydet
    cv2.imwrite(out_path, img)
    print(f"Kaydedildi: {out_path}")

def process_video(model, device, vid_path, out_path, conf_thres):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {vid_path}")
        return

    # Orijinal video ayarları
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter: mp4v codec ile .mp4 çıktısı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Toplam frame sayısı: {frame_count}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        # Inference
        results = model(frame, conf=conf_thres, device=device)[0]
        # Çizimler
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        # Output videoya yaz
        writer.write(frame)

        if idx % 100 == 0:
            print(f"İşlenen frame: {idx}/{frame_count}")

    cap.release()
    writer.release()
    print(f"Video kaydedildi: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLO ile foto/video tespiti ve çıktı kaydetme")
    parser.add_argument('--weights',   '-w', required=True, help='Model ağırlıkları (.pt)')
    parser.add_argument('--input',     '-i', required=True, help='Girdi dosyası (resim veya video)')
    parser.add_argument('--output',    '-o', required=False, help='Çıktı dosyası (varsayılan: aynı klasöre çıkış)')
    parser.add_argument('--conf',      '-c', type=float, default=0.25, help='Confidence threshold (0.0–1.0)')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Modeli yükle
    model = YOLO(args.weights)

    # Çıktı dosya yolunu belirle
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            out_path = f"{base}_detected{ext}"
        else:
            out_path = f"{base}_detected.mp4"

    # Girdi tipi kontrolü
    _, ext = os.path.splitext(args.input.lower())
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(model, device, args.input, out_path, args.conf)
    else:
        process_video(model, device, args.input, out_path, args.conf)

if __name__ == '__main__':
    main()
