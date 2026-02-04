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
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # Load BEV Calibration
    H = np.load(HOMOGRAPHY_PATH)
    ppm = np.load(PPM_PATH)
    
    print("🎬 ADAS System Started. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # ---------------------------------------------------------
        # 1. LANE DETECTION (With Sky Masking)
        # ---------------------------------------------------------
        lane_mask = lane_model.detect(frame)
        
        # Mask out Sky/Trees (Top 40%)
        sky_limit = int(h * 0.40) 
        lane_mask[0:sky_limit, :] = 1 
        
        # Draw Road (Green) on Main Video
        color_seg = np.zeros_like(frame)
        color_seg[lane_mask == 0] = [0, 255, 0]
        display_frame = cv2.addWeighted(display_frame, 0.8, color_seg, 0.2, 0)

        # ---------------------------------------------------------
        # 2. OBJECT DETECTION & RISK
        # ---------------------------------------------------------
        cars = obj_model.process_frame(frame)
        closest_dist = 999.0
        
        for car in cars:
            dist = car['dist_m']
            
            # Color Logic
            color = (0, 255, 0) # Green
            if dist < 30: color = (0, 165, 255) # Orange
            if dist < 15: color = (0, 0, 255) # Red
            
            # Draw Box on Main Video
            x1, y1, x2, y2 = car['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{dist:.1f}m", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if dist < closest_dist:
                closest_dist = dist

        # ---------------------------------------------------------
        # 3. SEPARATE MINIMAP WINDOW
        # ---------------------------------------------------------
        # Make the map larger since it's now its own window
        map_h, map_w = 600, 400 
        minimap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        
        # Draw Grid Lines (Every 10 meters)
        scale_map = 5.0 # Pixels per meter (Zoom level)
        for i in range(0, 100, 10):
            y_pos = int(map_h - 20 - (i * scale_map))
            if y_pos > 0:
                cv2.line(minimap, (0, y_pos), (map_w, y_pos), (50, 50, 50), 1)
        
        # Draw Ego Car (Us) at bottom center
        cv2.circle(minimap, (map_w//2, map_h-20), 15, (255, 255, 255), -1)
        cv2.putText(minimap, "EGO", (map_w//2 - 20, map_h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Draw Other Cars
        for car in cars:
            dx_m = car['lat_m']   # Meters left/right
            dy_m = car['dist_m']  # Meters ahead
            
            # Map X: Center + Deviation
            # Scale lateral movement by 20x to make lane changes obvious
            map_x = int((map_w // 2) + (dx_m * 20)) 
            
            # Map Y: Bottom - Distance
            map_y = int(map_h - 20 - (dy_m * scale_map))
            
            # Draw Red Dot if valid
            if 0 < map_y < map_h and 0 < map_x < map_w:
                cv2.circle(minimap, (map_x, map_y), 10, (0, 0, 255), -1)
                cv2.putText(minimap, f"{dy_m:.1f}m", (map_x+12, map_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ---------------------------------------------------------
        # 4. SHOW WINDOWS
        # ---------------------------------------------------------
        status = "SAFE"
        if closest_dist < 20: status = "WARNING"
            
        cv2.putText(display_frame, f"STATUS: {status}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show Main Video
        cv2.imshow("ADAS Camera View", display_frame)
        
        # Show Separate Map Window
        cv2.imshow("BEV Radar Map", minimap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()