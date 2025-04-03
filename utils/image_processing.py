from ultralytics import YOLO
import cv2
import base64

model = YOLO("yolov8n.pt") 

classNames = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"]

def determine_position(y_center, img_height):
  return "front" if y_center > img_height * 0.5 else "back"

def process_image(img):
  results = model(img, stream=True)
  vehicles = []

  for r in results:
    for box in r.boxes:
      cls = int(box.cls[0])

      if cls < 0 or cls >= len(classNames):
        continue

      label_name = classNames[cls]
      if label_name != "car":
        continue

      x1, y1, x2, y2 = map(int, box.xyxy[0])
      confidence = float(box.conf[0])
      center_x = (x1 + x2) / 2
      center_y = (y1 + y2) / 2
      area = (x2 - x1) * (y2 - y1)
      position = determine_position(center_y, img.shape[0])

      if confidence < 0.5:
        continue

      vehicles.append({
          "type": classNames[cls],
          "confidence": confidence,
          "boundingBox": [x1, y1, x2, y2],
          "center": [center_x, center_y],
          "area": area,
          "position": position
      })

      color = (0, 255, 0) if position == "front" else (255, 0, 255)
      cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
      label = f"{classNames[cls]} {confidence:.2f}"
      cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

  _, buffer = cv2.imencode('.jpg', img)
  img_str = base64.b64encode(buffer).decode('utf-8')

  return {
        "vehicles": vehicles,
        "processedImage": f"data:image/jpeg;base64,{img_str}"
  }