import cv2
import os
from ultralytics import YOLO


model = YOLO("best.pt")

input_image = "data/emergency.jpeg"
output_dir = "data"
output_image = os.path.join(output_dir, "emergency_detected.jpg")

img = cv2.imread(input_image)
if img is None:
    print("Gambar tidak ditemukan")
    exit()

results = model(img, conf=0.25)
annotated = results[0].plot()

cv2.imwrite(output_image, annotated)

print(f"Hasil deteksi disimpan di: {output_image}")
