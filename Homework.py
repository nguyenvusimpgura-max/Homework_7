import os
import cv2
import numpy as np
import tensorflow as tf


MODEL_FILENAME = 'identity_recognition_model.h5'
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, MODEL_FILENAME)
print(f"[INFO] Attempting to load model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    cwd_path = os.path.join(os.getcwd(), MODEL_FILENAME)
    print(f"[WARN] Not found at script location, checking CWD: {cwd_path}")
    if os.path.exists(cwd_path):
        MODEL_PATH = cwd_path
    else:
        raise FileNotFoundError(
            f"Model file not found. Tried:\n  {MODEL_PATH}\n  {cwd_path}")

model = tf.keras.models.load_model(MODEL_PATH)

try:
    input_shape = model.input_shape  # (None, width, height, channels)
    TARGET_SIZE = (int(input_shape[1]), int(input_shape[2]))
except Exception:
    TARGET_SIZE = (224, 224)
print(f"[INFO] Using target image size: {TARGET_SIZE}")

# Face detector used to crop the face before prediction.
face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
)
if face_cascade.empty():
    raise RuntimeError('Could not load haarcascade_frontalface_default.xml')


CLASS_NAMES = [
    'NGUYEN HUU TOAN',
    'NGUYEN KHAC LUU VU',
    'NGUYEN TRONG MINH',
    'STRANGER',
    'TRAN MINH HOANG',
    'TRAN THE DANG KHOA'
]

CONFIDENCE_THRESHOLD = 0.75
MARGIN = 0.20

index_to_class = {index: class_name for index, class_name in enumerate(CLASS_NAMES)}

cap = cv2.VideoCapture(0) # Số 0 nghĩa là mở webcam mặc định của laptop

print("[INFO] Đang bật Webcam... Bấm phím 'q' trên bàn phím để THOÁT.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Không thể kết nối với webcam.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    h, w, _ = frame.shape

    for (x, y, face_w, face_h) in faces:
        # Mở rộng khung cắt để lấy thêm tóc và cằm
        margin = int(face_w * MARGIN)
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(w, x + face_w + margin)
        y_max = min(h, y + face_h + margin)

        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, TARGET_SIZE)
        face_normalized = face_resized / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)

        predictions = model.predict(face_batch, verbose=0)[0]
        max_prob = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))

        predicted_class_name = index_to_class.get(class_idx, 'STRANGER')
        if max_prob < CONFIDENCE_THRESHOLD or predicted_class_name == 'STRANGER':
            label = 'Stranger'
            color = (0, 0, 255)
        else:
            label = f"{predicted_class_name} ({max_prob:.2f})"
            color = (0, 255, 0)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            frame,
            label,
            (x_min, max(30, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    if len(faces) == 0:
        cv2.putText(
            frame,
            'No face detected',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("AI Face Recognition Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()