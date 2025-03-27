from ultralytics import YOLO
import cv2
import math
import sys

def process_image(img, model, classNames):
  results = model(img, stream=True)

  front_car_box = None
  max_y2 = -1

  for r in results:
    boxes = r.boxes
    for box in boxes:
      cls = int(box.cls[0])

      if classNames[cls] == "car":
        continue

      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
      confidence = math.ceil((box.conf[0] * 100))/100
      print("Confidence --->", confidence)
      print("Detected: ", classNames[cls])

      if y2 > max_y2:
        max_y2 = y2
        front_car_box = (x1, y1, x2, y2)
  if front_car_box:
        x1, y1, x2, y2 = front_car_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, "Front Car", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print("Front car detected at:", front_car_box)

  return img

if __name__ == "__main__":
  model = YOLO("yolo-Weights/yolov8n.pt")

  classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
  ]

  if len(sys.argv) > 1:
    for file_path in sys.argv[1:]:
      img = cv2.imread(file_path)
      if img is None:
        print(f"Error: Could not find image {file_path}")
        continue

      result_img = process_image(img, model, classNames)
      cv2.imshow("Detection", result_img)
      print(f"Press any key to continue after viewing {file_path}")
      cv2.waitKey(0)
    cv2.destroyAllWindows()
  else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      print("Error: Could not open camera")
      exit()

    while True:
      ret, img = cap.read()
      if not ret:
        print("Error: Could not read frame.")
        break

      result_img = process_image(img, model, classNames)
      cv2.imshow("Webcam", result_img)

      if cv2.waitKey(1) == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()