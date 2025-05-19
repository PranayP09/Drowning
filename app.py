import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request

# Download YOLO weights if not present
def download_weights():
    weights_path = "yolov3.weights"
    if not os.path.exists(weights_path):
        st.info("Downloading YOLO weights...")
        url = "https://pjreddie.com/media/files/yolov3.weights"
        urllib.request.urlretrieve(url, weights_path)
        st.success("Weights downloaded successfully!")
    return weights_path

# Load YOLO model
def load_yolo():
    weights = download_weights()
    config = "yolov3.cfg"
    labels = "yolov3.txt"
    net = cv2.dnn.readNet(weights, config)
    classes = []
    with open(labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Perform detection
def detect_drowning(net, output_layers, classes, frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Streamlit UI
st.title("Drowning Detection App 🌊")
st.write("Upload a video to detect drowning using AI.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    net, classes, output_layers = load_yolo()

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = detect_drowning(net, output_layers, classes, frame)
        stframe.image(result_frame, channels="BGR")

    cap.release()
    st.success("Video processing complete.")
