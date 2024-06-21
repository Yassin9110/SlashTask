import cv2
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pandas as pd


model = YOLO('yolov8s.pt')


def detect_objects(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img)  
    detections = results[0].boxes.data.cpu().numpy()  
    return detections, results[0].names


def draw_boxes(image, detections, names):
    img = np.array(image)
    for box in detections:
        x1, y1, x2, y2, confidence, class_id = box[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names[int(class_id)]}: {confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return img

# Streamlit app setup
st.title("Object Detection")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        detections, names = detect_objects(image)
        image_with_boxes = draw_boxes(image, detections, names)
        st.image(image_with_boxes, caption="Detected Objects", use_column_width=True)
        
        # Create a DataFrame for the detection details
        detection_details = []
        for box in detections:
            x1, y1, x2, y2, confidence, class_id = box[:6]
            details = {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": confidence,
                "class": class_id,
                "name": names[class_id]
            }
            detection_details.append(details)
        
        
        df = pd.DataFrame(detection_details)
        st.dataframe(df)
