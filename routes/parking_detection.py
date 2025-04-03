from fastapi import APIRouter, UploadFile, HTTPException, File
import numpy as np
import cv2

from utils.image_processing import process_image

router = APIRouter()

@router.post("/parking-detection")
async def detect_parking(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image")

  contents = await file.read()
  nparr = np.frombuffer(contents, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if img is None:
    raise HTTPException(status_code=400, detail="Invalid image")

  result = process_image(img)
  front_count = sum(1 for v in result["vehicles"] if v["position"] == "front")
  back_count = sum(1 for v in result["vehicles"] if v["position"] == "back")

  return {
    "totalVehicles": len(result["vehicles"]),
    "frontVehicles": front_count,
    "backVehicles": back_count,
    "vehicles": result["vehicles"],
    "processedImage": result["processedImage"]
  }