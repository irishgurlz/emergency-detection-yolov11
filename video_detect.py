import cv2
import os
from ultralytics import YOLO


model = YOLO("best.pt")

input_video = "data/video-emergency.mp4"
output_video = "data/video-emergency-detected.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Video tidak bisa dibuka")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)
    annotated = results[0].plot()

    out.write(annotated)

cap.release()
out.release()

print(f"Hasil deteksi video disimpan di: {output_video}")
