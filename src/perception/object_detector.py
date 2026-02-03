import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        print(f"⏳ Loading YOLOv8 ({model_name})...")
        
        # 1. Load Model (Ultralytics handles device selection automatically)
        self.model = YOLO(model_name)
        
        # 2. Define Classes we care about (COCO Dataset IDs)
        # 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7]
        
        print("✅ YOLOv8 Loaded Successfully.")

    def detect(self, image):
        # Run inference
        results = self.model(image, verbose=False)[0]
        
        detections = []
        
        # Process results
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in self.target_classes and conf > 0.4:
                # Get coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": results.names[cls_id],
                    "conf": conf
                })
        
        return detections

    def visualize(self, image, detections):
        vis_img = image.copy()
        
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            label = f"{d['class']} {d['conf']:.2f}"
            
            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw Label Background
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 0, 255), -1)
            
            # Draw Text
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return vis_img

if __name__ == "__main__":
    img_path = "assets/calibration_target.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print("❌ Error: Image not found.")
    else:
        detector = ObjectDetector()
        
        # Detect
        objects = detector.detect(img)
        print(f"✅ Detected {len(objects)} vehicles.")
        
        # Visualize
        result = detector.visualize(img, objects)
        cv2.imshow("YOLOv8 Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()