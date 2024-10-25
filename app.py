import streamlit as st
import cv2
import numpy as np
import requests

st.title("YOLO Video Stream")

stframe = st.empty()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    response = requests.post("http://localhost:8000/process_video/", files={"file": img})
    img = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    stframe.image(img, channels="BGR")

cap.release()
