# routes/license_plate.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

router = APIRouter()

PLATE_REGEX = re.compile(r"(?i)^[A-Z]{2}[- ]?\d{5}$")

reader = easyocr.Reader(['en'], gpu=False) 
model = YOLO("yolov8n.pt")  

@router.post("/license-plate")
async def detect_license_plate(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    results = model(img)
    detections = results[0].boxes.data.cpu().numpy()

    plates = []

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        plate_crop = gray[y1:y2, x1:x2]

        ocr_results = reader.readtext(plate_crop)
        for _, text, _ in ocr_results:
            cleaned = text.replace(" ", "").replace("-", "").replace(":", "")
            if PLATE_REGEX.match(cleaned):
                plates.append(cleaned.upper())

    return {"license_plates": plates}
