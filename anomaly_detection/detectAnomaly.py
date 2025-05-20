# detect_anomaly.py

import cv2
import numpy as np
import tensorflow as tf

# Aynı sabitler
IMG_HEIGHT = 64
IMG_WIDTH  = 64
CLASS_LABELS = ['Abuse','Arrest','Arson','Assault','Burglary',
                'Explosion','Fighting','Normal','RoadAccidents',
                'Robbery','Shooting','Shoplifting','Stealing','Vandalism']

preprocess_fun = tf.keras.applications.densenet.preprocess_input

def preprocess_frame(frame):
    # BGR -> RGB, resize, normalize ve DenseNet ön-işlemi
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return preprocess_fun(img)

if __name__ == "__main__":
    # 1) Modeli yükle
    model = tf.keras.models.load_model("anomaly_detector.h5", compile=False)

    # 2) Video kaynak ve çıktı ayarları
    cap = cv2.VideoCapture("input_video.mp4")  # videonun adı
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prep  = preprocess_frame(frame)
        pred  = model.predict(np.expand_dims(prep, 0), verbose=0)[0]
        idx   = np.argmax(pred)
        label = CLASS_LABELS[idx]
        prob  = pred[idx]

        # “Normal” dışındaysa anomali say
        if label != "Normal":
            text = f"Anomaly: {label} ({prob*100:.1f}%)"
            cv2.putText(frame, text, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        out.write(frame)
        # Gerçek zamanlı izlemek isterseniz:
        # cv2.imshow("Anomaly", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print("İşleme tamamlandı, çıktı: output.mp4")
