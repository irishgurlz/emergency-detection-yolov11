import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Emergency Vehicle Detection")

mode = st.radio(
    "Pilih input",
    ["Image", "Video", "Webcam"]
)

os.makedirs("data", exist_ok=True)

if mode == "Image":
    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_image.read())

        img = cv2.imread(tfile.name)
        results = model(img, conf=0.25)
        annotated = results[0].plot()

        output_path = "data/image_detected.jpg"
        cv2.imwrite(output_path, annotated)

        st.image(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download hasil image",
                f,
                file_name="image_detected.jpg"
            )

elif mode == "Video":
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = "data/video_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        status_text = st.empty()
        progress_bar = st.progress(0.0)

        frame_idx = 0
        status_text.write("Memproses video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.25)
            annotated = results[0].plot()
            out.write(annotated)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.write(f"Frame {frame_idx} / {total_frames}")

        cap.release()
        out.release()

        status_text.write("Selesai")

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download hasil video",
                f,
                file_name="video_detected.mp4"
            )

elif mode == "Webcam":
    run = st.checkbox("Nyalakan Webcam")
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)
        annotated = results[0].plot()

        stframe.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

    cap.release()
