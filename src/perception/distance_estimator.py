import cv2
import numpy as np
import os
from src.perception.object_detector import ObjectDetector

class DistanceEstimator:
    def __init__(self, homography_path, ppm_path):
        # Load Calibration
        if not os.path.exists(homography_path):
            raise FileNotFoundError("Calibration not found!")
            
        self.H = np.load(homography_path)
        self.ppm = np.load(ppm_path)
        
        # Load Detector (Lower threshold to 0.3 to catch more cars)
        self.detector = ObjectDetector() 
        # Hack: Manually lower the threshold in the detector instance if needed
        # (Or we can just filter differently here)

    def process_frame(self, image):
        # 1. Detect Objects
        detections = self.detector.detect(image)
        
        results = []
        
        # 2. Convert Boxes to Real-World Distances
        for car in detections:
            x1, y1, x2, y2 = car['bbox']
            
            # Find the "Contact Point" (Bottom-Center of the box)
            # This represents where the tires touch the road
            cx = (x1 + x2) // 2
            cy = y2 
            
            # 3. Project to BEV (The Matrix Math)
            # Reshape for OpenCV: (1 point, 1, 2 coordinates)
            point_src = np.array([[[cx, cy]]], dtype=np.float32)
            point_dst = cv2.perspectiveTransform(point_src, self.H)
            
            bev_x = point_dst[0][0][0]
            bev_y = point_dst[0][0][1]
            
            # 4. Calculate Distance relative to "Ego Car"
            # In our BEV setup (from Phase 1), the camera is at the bottom-center
            # BEV Image Size was 400x600.
            # So Ego position is roughly (200, 600)
            
            ego_x = 200 # Center width
            ego_y = 600 # Bottom height
            
            # Distance in Pixels
            dx_px = bev_x - ego_x
            dy_px = ego_y - bev_y # Y is inverted (0 is top)
            
            # Distance in Meters
            dist_long_m = dy_px / self.ppm # Longitudinal (Forward) distance
            dist_lat_m = dx_px / self.ppm  # Lateral (Side) distance
            
            # Filter: Ignore cars "behind" us or impossible detections
            if dist_long_m > 0:
                car['dist_m'] = dist_long_m
                car['lat_m'] = dist_lat_m
                results.append(car)
                
        return results

    def visualize(self, image, results):
        vis_img = image.copy()
        
        for car in results:
            x1, y1, x2, y2 = car['bbox']
            dist = car['dist_m']
            
            # Color logic: Red if close (< 20m), Green if far
            color = (0, 0, 255) if dist < 20 else (0, 255, 0)
            
            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw Distance Text
            label = f"{dist:.1f}m"
            cv2.putText(vis_img, label, (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return vis_img

if __name__ == "__main__":
    img_path = "assets/calibration_target.jpg"
    img = cv2.imread(img_path)
    
    estimator = DistanceEstimator("configs/homography_matrix.npy", "configs/pixels_per_meter.npy")
    
    # Run Logic
    cars_with_distance = estimator.process_frame(img)
    
    # Print Results
    print(f"\n✅ Analyzed {len(cars_with_distance)} vehicles:")
    for car in cars_with_distance:
        print(f"   - {car['class']}: {car['dist_m']:.2f} meters ahead")
    
    # Show Image
    final_img = estimator.visualize(img, cars_with_distance)
    cv2.imshow("Distance Estimation", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()