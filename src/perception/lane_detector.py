import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class LaneDetector:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-cityscapes-512-1024"):
        print(f"⏳ Loading Lane Detector: {model_name}...")
        
        # 1. Detect Device (MPS for Mac M2, CUDA for Nvidia, CPU otherwise)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ Using Device: Apple MPS (GPU accelerated)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✅ Using Device: CUDA (GPU accelerated)")
        else:
            self.device = torch.device("cpu")
            print("⚠️ Using Device: CPU (Inference might be slow)")

        # 2. Load Model & Processor
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode (no training)
        
        print("✅ Model Loaded Successfully.")

    def detect(self, image):
        # 1. Preprocess the image (Resize & Normalize for the AI)
        # return_tensors="pt" gives us PyTorch tensors
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Run Inference (The "Thinking" part)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. Post-process (Scale output back to original image size)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2], # (Height, Width) of original image
            mode="bilinear",
            align_corners=False,
        )

        # 4. Get the Class ID for each pixel
        # The model outputs probabilities for all classes. We take the highest one.
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        
        # Move result back to CPU for OpenCV to use
        return pred_seg.cpu().numpy().astype(np.uint8)

    def visualize(self, image, mask):
        # Cityscapes Palette (Simplified)
        # Class 0 = Road, Class 1 = Sidewalk...
        
        # Create a blank color image
        color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Color the Road (Class 0) Green
        color_seg[mask == 0] = [0, 255, 0] 
        
        # Blend original image with the mask
        # 0.6 * Image + 0.4 * Mask
        vis = cv2.addWeighted(image, 0.6, color_seg, 0.4, 0)
        return vis

if __name__ == "__main__":
    # Test on the calibration frame
    img_path = "assets/calibration_target.jpg"
    
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Run extract_frame.py first!")
        exit()

    detector = LaneDetector()
    mask = detector.detect(img)
    result = detector.visualize(img, mask)

    cv2.imshow("AI Segmentation Output (Green = Road)", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()