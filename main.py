from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import sqlite3
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import math
import cvzone
import time

app = FastAPI()

# Connect to database
conn = sqlite3.connect('my_database.db')

# Create a table
conn.execute('''CREATE TABLE IF NOT EXISTS my_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image BLOB
);''')

model = YOLO("../Yolo-Weights/best.pt")
classNames = ["gun", "fire", "big_gun"]

def process_frame(img):
    results = model(img, stream=True)
    now = datetime.datetime.now()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = classNames[cls]
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            if current_class == "fire":
                name = "Dangerous Fire"
                _, jpeg_data = cv2.imencode('.jpg', img)
                conn.execute("INSERT INTO my_table (name, time, image) VALUES (?, ?, ?)",
                             (name, now, sqlite3.Binary(jpeg_data)))
                conn.commit()
    return img

@app.get("/process_video/")
async def process_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        img = process_frame(img)
        _, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()

@app.get("/")
def root():
    return {"message": "FastAPI with YOLO is running!"}
