import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak terdeteksi")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    annotated = results[0].plot()

    cv2.imshow("Emergency Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
