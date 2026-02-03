import cv2
import numpy as np
import os
from src.perception.lane_detector import LaneDetector

class BEVLaneFitter:
    def __init__(self, homography_path, ppm_path):
        # Load Calibration
        if not os.path.exists(homography_path):
            raise FileNotFoundError("Calibration not found. Run camera_calibration.py first!")
        
        self.H = np.load(homography_path)
        self.ppm = np.load(ppm_path)
        
        self.BEV_WIDTH = 400
        self.BEV_HEIGHT = 600
        self.detector = LaneDetector()

    def process_frame(self, image):
        # 1. Get AI Mask
        raw_mask = self.detector.detect(image)
        lane_mask = np.zeros_like(raw_mask)
        lane_mask[raw_mask == 0] = 255 
        
        # --- NEW: Mask out the Car Hood ---
        h, w = lane_mask.shape
        # Black out the bottom 10% of the image to remove the dashboard/hood
        cv2.rectangle(lane_mask, (0, int(h * 0.90)), (w, h), (0, 0, 0), -1)
        
        # 2. Warp to BEV
        bev_mask = cv2.warpPerspective(lane_mask, self.H, (self.BEV_WIDTH, self.BEV_HEIGHT))
        
        # 3. Extract Edges
        edges = cv2.Canny(bev_mask, 50, 150)
        
        # 4. Find Lane Deviation
        midpoint = self.BEV_WIDTH // 2
        
        # Get all white pixel coordinates
        y_idxs, x_idxs = edges.nonzero()
        
        # If we find edges
        if len(y_idxs) > 0:
            # We only care about the bottom half of the BEV (closest to car)
            # This reduces noise from the far horizon
            valid_indices = y_idxs > (self.BEV_HEIGHT * 0.5)
            
            if np.any(valid_indices):
                filtered_x = x_idxs[valid_indices]
                filtered_y = y_idxs[valid_indices]
                
                # Split into Left and Right relative to center
                left_pixels = filtered_x[filtered_x < midpoint]
                right_pixels = filtered_x[filtered_x >= midpoint]
                
                l_pos = midpoint
                r_pos = midpoint
                
                # Find the average position of the left line
                if len(left_pixels) > 0:
                    l_pos = np.mean(left_pixels)
                    
                # Find the average position of the right line
                if len(right_pixels) > 0:
                    r_pos = np.mean(right_pixels)
                
                # Calculate the center of the lane
                lane_center = (l_pos + r_pos) / 2
                
                # Deviation: (Car Center) - (Lane Center)
                deviation_px = midpoint - lane_center
                deviation_m = deviation_px / self.ppm
                
                # --- Visualization ---
                # Draw the lines we found
                debug_img = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
                cv2.line(debug_img, (int(l_pos), 0), (int(l_pos), self.BEV_HEIGHT), (0, 0, 255), 2) # Red = Left
                cv2.line(debug_img, (int(r_pos), 0), (int(r_pos), self.BEV_HEIGHT), (255, 0, 0), 2) # Blue = Right
                cv2.line(debug_img, (int(lane_center), 0), (int(lane_center), self.BEV_HEIGHT), (0, 255, 0), 2) # Green = Center
                cv2.line(debug_img, (midpoint, 0), (midpoint, self.BEV_HEIGHT), (255, 255, 255), 1) # White = Car
                
                return debug_img, deviation_m

        return bev_mask, 0.0

if __name__ == "__main__":
    img_path = "assets/calibration_target.jpg"
    img = cv2.imread(img_path)
    
    fitter = BEVLaneFitter("configs/homography_matrix.npy", "configs/pixels_per_meter.npy")
    
    bev_result, offset = fitter.process_frame(img)
    
    print(f"\n✅ RE-CALCULATED Lane Deviation: {offset:.2f} meters")
    
    cv2.imshow("Debug: Red=Left, Blue=Right, Green=Center", bev_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()