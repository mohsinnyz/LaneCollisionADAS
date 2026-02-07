import cv2
import numpy as np
import time
from src.perception.lane_detector import LaneDetector
from src.perception.distance_estimator import DistanceEstimator

def main():
    # --- CONFIGURATION ---
    VIDEO_PATH = "assets/test_video.mp4"
    HOMOGRAPHY_PATH = "configs/homography_matrix.npy"
    PPM_PATH = "configs/pixels_per_meter.npy"
    
    # Load Models
    lane_model = LaneDetector()
    obj_model = DistanceEstimator(HOMOGRAPHY_PATH, PPM_PATH)
    
    # Video Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Load BEV Calibration
    H = np.load(HOMOGRAPHY_PATH)
    ppm = np.load(PPM_PATH)
    
    print("🎬 ADAS System Started. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Prepare Original Frame (Resize for display)
        # We process the full resolution frame but display smaller versions
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # ---------------------------------------------------------
        # 2. LANE DETECTION (With Sky Masking)
        # ---------------------------------------------------------
        lane_mask = lane_model.detect(frame)
        
        # Mask out Sky/Trees (Top 40%)
        sky_limit = int(h * 0.40) 
        lane_mask[0:sky_limit, :] = 1 
        
        # Draw Road (Green) on ADAS Frame
        color_seg = np.zeros_like(frame)
        color_seg[lane_mask == 0] = [0, 255, 0]
        display_frame = cv2.addWeighted(display_frame, 0.8, color_seg, 0.2, 0)

        # ---------------------------------------------------------
        # 3. OBJECT DETECTION & RISK
        # ---------------------------------------------------------
        cars = obj_model.process_frame(frame)
        closest_dist = 999.0
        
        for car in cars:
            dist = car['dist_m']
            
            # Color Logic
            color = (0, 255, 0) # Green
            if dist < 30: color = (0, 165, 255) # Orange
            if dist < 15: color = (0, 0, 255) # Red
            
            # Draw Box on ADAS Frame
            x1, y1, x2, y2 = car['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{dist:.1f}m", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if dist < closest_dist:
                closest_dist = dist

        # ---------------------------------------------------------
        # 4. BEV MAP WINDOW
        # ---------------------------------------------------------
        map_h, map_w = 600, 400 
        minimap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        
        # Draw Grid Lines
        scale_map = 5.0 
        for i in range(0, 100, 10):
            y_pos = int(map_h - 20 - (i * scale_map))
            if y_pos > 0:
                cv2.line(minimap, (0, y_pos), (map_w, y_pos), (50, 50, 50), 1)
        
        # Draw Ego Car
        cv2.circle(minimap, (map_w//2, map_h-20), 15, (255, 255, 255), -1)
        cv2.putText(minimap, "EGO", (map_w//2 - 20, map_h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Draw Other Cars on Map
        for car in cars:
            dx_m = car['lat_m']
            dy_m = car['dist_m']
            
            map_x = int((map_w // 2) + (dx_m * 20)) 
            map_y = int(map_h - 20 - (dy_m * scale_map))
            
            if 0 < map_y < map_h and 0 < map_x < map_w:
                cv2.circle(minimap, (map_x, map_y), 10, (0, 0, 255), -1)
                cv2.putText(minimap, f"{dy_m:.1f}m", (map_x+12, map_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ---------------------------------------------------------
        # 5. SHOW ALL 3 WINDOWS
        # ---------------------------------------------------------
        status = "SAFE"
        if closest_dist < 20: status = "WARNING"
        cv2.putText(display_frame, f"STATUS: {status}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize windows to fit screen nicely (e.g. 640x360 for 16:9 video)
        # Resizing purely for display purposes, does not affect detection accuracy
        display_w, display_h = 640, 360
        
        # 1. Original Raw Video
        cv2.imshow("Original Input", cv2.resize(frame, (display_w, display_h)))
        
        # 2. ADAS Processing
        cv2.imshow("ADAS Output", cv2.resize(display_frame, (display_w, display_h)))
        
        # 3. Map (Keep original size or resize if needed)
        cv2.imshow("Radar Map", minimap)

        # Move windows so they don't overlap
        # Adjust these X, Y values based on your screen resolution
        cv2.moveWindow("Original Input", 0, 0)
        cv2.moveWindow("ADAS Output", display_w + 10, 0)
        cv2.moveWindow("Radar Map", (display_w * 2) + 20, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()