import cv2
import os

def extract_first_frame(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"✅ Success! Frame saved to {output_path}")
    else:
        print("❌ Error: Failed to read video. Check file format.")
    
    cap.release()

if __name__ == "__main__":
    # Ensure you have a video in assets/ named test_video.mp4
    extract_first_frame("assets/test_video.mp4", "assets/calibration_target.jpg")