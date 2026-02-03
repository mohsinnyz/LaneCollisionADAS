import cv2
import numpy as np
import os

class CalibrationTool:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        self.clone = self.image.copy()
        self.points = []
        self.window_name = "Calibration: Click 4 Points (BL, TL, TR, BR)"
        
        # --- CONFIGURATION ---
        self.LANE_WIDTH_METERS = 3.7  # Standard US Highway width
        self.BEV_WIDTH = 400          # Output BEV image width
        self.BEV_HEIGHT = 600         # Output BEV image height
        
    def click_event(self, event, x, y, flags, params):
        # Listen for Left-Clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                # Draw a red circle and the number
                cv2.circle(self.clone, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.clone, str(len(self.points)), (x+10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.clone)

    def run(self):
        cv2.imshow(self.window_name, self.image)
        cv2.setMouseCallback(self.window_name, self.click_event)
        
        print("\n--- INSTRUCTIONS ---")
        print("1. Click the BOTTOM-LEFT of a lane line.")
        print("2. Click the TOP-LEFT of the same lane line.")
        print("3. Click the TOP-RIGHT of the other lane line.")
        print("4. Click the BOTTOM-RIGHT of the other lane line.")
        print("   (Imagine drawing a rectangle on the road floor)")
        print("Press 'c' to Calculate, 'r' to Reset, 'q' to Quit.\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 'c' key: Calculate
            if key == ord("c") and len(self.points) == 4:
                self.compute_homography()
                break
            # 'r' key: Reset
            elif key == ord("r"):
                self.points = []
                self.clone = self.image.copy()
                cv2.imshow(self.window_name, self.clone)
            # 'q' key: Quit
            elif key == ord("q"):
                break
        
        cv2.destroyAllWindows()

    def compute_homography(self):
        src_pts = np.float32(self.points)
        
        # --- DESTINATION MAPPING ---
        # We map the pixels we clicked to a perfect rectangle in the new image.
        # We arbitrarily decide that the lane is 100 pixels wide in our new view.
        target_lane_width_px = 100 
        
        # Calculate PPM (Pixels Per Meter) based on known lane width (3.7m)
        ppm = target_lane_width_px / self.LANE_WIDTH_METERS
        
        # Center the lane in our 400x600 output image
        cx = self.BEV_WIDTH // 2
        
        # Define the 4 points in the "Bird's Eye View"
        # We assume the user clicked a segment roughly 30 meters long
        # (The length doesn't matter for width calibration, but helps visualization)
        
        # Bottom-Left (x, y)
        bl = [cx - target_lane_width_px/2, self.BEV_HEIGHT - 50]
        # Top-Left (x, y)
        tl = [cx - target_lane_width_px/2, self.BEV_HEIGHT - 50 - (30 * ppm)]
        # Top-Right (x, y)
        tr = [cx + target_lane_width_px/2, self.BEV_HEIGHT - 50 - (30 * ppm)]
        # Bottom-Right (x, y)
        br = [cx + target_lane_width_px/2, self.BEV_HEIGHT - 50]
        
        dst_pts = np.float32([bl, tl, tr, br])
        
        # Calculate the Matrix H
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Save to configs folder
        np.save("configs/homography_matrix.npy", H)
        np.save("configs/pixels_per_meter.npy", ppm)
        
        print(f"✅ Saved 'homography_matrix.npy'")
        print(f"✅ Calculated Scale: {ppm:.2f} pixels/meter")
        
        # Show result
        warped = cv2.warpPerspective(self.image, H, (self.BEV_WIDTH, self.BEV_HEIGHT))
        cv2.imshow("Result: Bird's Eye View", warped)
        print("Press any key to close...")
        cv2.waitKey(0)

if __name__ == "__main__":
    if not os.path.exists("assets/calibration_target.jpg"):
        print("❌ Error: assets/calibration_target.jpg not found.")
    else:
        tool = CalibrationTool("assets/calibration_target.jpg")
        tool.run()