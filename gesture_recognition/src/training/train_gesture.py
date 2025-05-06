import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


DATA_DIR = '/Users/erinc/GitHub/SecurityOfficer/gesture_recognition/data/raw'
SEQ_LEN = 30
FEATURE_DIM = 63  # 21 landmark x 3

def load_data():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print("DEBUG: bulduğum CSV dosyaları:", files)    # ← ekledik
    sequences, labels = [], []
    for file in files:
        df = pd.read_csv(file, header=None)
        print(f"DEBUG: {file} içindeki satır sayısı:", len(df))  # ← ekledik
        for _, row in df.iterrows():
            label = row[0]
            vector = row[1:].values.astype(float)
            seq = np.array_split(vector, len(vector) / FEATURE_DIM)
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

# Model tanımı
def build_model(n_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQ_LEN, FEATURE_DIM)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X, y = load_data()
    labels = sorted(list(set(y)))
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_idx = np.array([label_map[label] for label in y])
    y_cat = to_categorical(y_idx, num_classes=len(labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = build_model(n_classes=len(labels))
    model.summary()
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
    model.save('gesture_lstm.h5')
    print("Model saved as gesture_lstm.h5")