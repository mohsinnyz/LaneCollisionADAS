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
    
    # Risk Thresholds
    TTC_THRESHOLD_DANGER = 2.5 # Seconds (Alert if collision < 2.5s)
    
    # --- INITIALIZATION ---
    print(f"🎬 Opening Video: {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Get video FPS for time calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    dt = 1.0 / fps # Time per frame (seconds)
    
    # Load Models
    lane_model = LaneDetector() # Phase 2
    obj_model = DistanceEstimator(HOMOGRAPHY_PATH, PPM_PATH) # Phase 3 (includes Matrix)
    
    # State Variables (Memory)
    prev_distance = None
    smoothed_ttc = 99.0 # Start with "Safe"
    
    # --- MAIN LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # 1. Resize for speed (optional, SegFormer likes 512x512 but we keep aspect)
        # We'll work on the original frame for accuracy
        display_frame = frame.copy()
        
        # 2. RUN LANE DETECTION (Drift)
        # (Using the detector raw method for speed, visualize manually)
        lane_mask = lane_model.detect(frame)
        
        # Color the road green (Visual only)
        color_seg = np.zeros_like(frame)
        color_seg[lane_mask == 0] = [0, 255, 0]
        display_frame = cv2.addWeighted(display_frame, 0.7, color_seg, 0.3, 0)
        
        # 3. RUN OBJECT DETECTION (Distance)
        # We get a list of cars with 'dist_m' calculated
        cars = obj_model.process_frame(frame)
        
        # 4. CALCULATE RISK (Time-to-Collision)
        # Strategy: Find the CLOSEST car in front of us.
        min_dist = 999.0
        target_car = None
        
        for car in cars:
            if car['dist_m'] < min_dist:
                min_dist = car['dist_m']
                target_car = car
        
        # Physics Logic
        risk_color = (0, 255, 0) # Green (Safe)
        risk_text = "SAFE"
        
        if target_car:
            current_dist = target_car['dist_m']
            
            # Calculate Relative Speed (Distance Change / Time)
            if prev_distance is not None:
                # How much closer did it get?
                delta_dist = prev_distance - current_dist
                
                # Speed in meters/second
                rel_speed = delta_dist / dt 
                
                # Only calculate TTC if we are closing in (speed > 0)
                if rel_speed > 0.1: 
                    ttc = current_dist / rel_speed
                    
                    # Smooth the TTC value so it doesn't flicker
                    smoothed_ttc = (0.9 * smoothed_ttc) + (0.1 * ttc)
                else:
                    smoothed_ttc = 99.0 # Valid but pulling away
            
            # Update memory
            prev_distance = current_dist
            
            # --- ALERT LOGIC ---
            if smoothed_ttc < TTC_THRESHOLD_DANGER:
                risk_color = (0, 0, 255) # RED
                risk_text = f"COLLISION WARNING: {smoothed_ttc:.1f}s"
            elif smoothed_ttc < 5.0:
                risk_color = (0, 165, 255) # Orange
                risk_text = f"Caution: {smoothed_ttc:.1f}s"
            
            # Draw Box & Info on the Target Car
            x1, y1, x2, y2 = target_car['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), risk_color, 3)
            cv2.putText(display_frame, f"{current_dist:.1f}m", (x1, y2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

        # 5. Draw Dashboard
        # Top Banner
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(display_frame, f"System Status: {risk_text}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, risk_color, 2)
        
        # Show Video
        cv2.imshow("ADAS Prototype (MacBook M2)", display_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()