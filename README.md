# 🚗 LaneCollisionADAS
### Advanced Driver Assistance System using Computer Vision & AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-yellow)

**LaneCollisionADAS** is a Python-based Advanced Driver Assistance System (ADAS) that leverages state-of-the-art computer vision and deep learning techniques to enhance road safety. It processes standard dashcam footage to provide real-time lane segmentation, vehicle detection, distance estimation, and collision warnings.

---

## 🌟 Key Features

*   **🛣️ Semantic Lane Segmentation**: Uses the **Segformer** transformer model (HuggingFace) to accurately identify drivable road areas, robust to shadows and lighting changes.
*   **🚗 Object Detection & Tracking**: Implements **YOLOv8** to detect vehicles (cars, trucks, buses, motorcycles) with high precision.
*   **📏 Distance Estimation**: Utilizes **Planar Homography** and camera calibration to estimate the real-world distance (in meters) of vehicle contacts points.
*   **🛰️ Bird's Eye View (BEV) Radar**: Projects the road scene into a top-down 2D map, visualizing ego-vehicle position relative to surrounding traffic.
*   **⚠️ Collision Warning System**: actively monitors headway distance and alerts drivers with color-coded visual warnings (Green → Orange → Red).

## 🎥 Demo

See the system in action:

[![Watch the Demo](https://img.shields.io/badge/Watch_Demo-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohsinnyz/recent-activity/all/)


---


## 🛠️ System Architecture

The pipeline processes each video frame through the following stages:

1.  **Input Frame**: Capture video from a file or camera stream.
2.  **Perception Layer**:
    *   **Lane detection**: `Segformer` masks the road surface. top 40% of the image (sky/trees) is filtered out.
    *   **Object Detection**: `YOLOv8` provides bounding boxes for vehicles.
3.  **Geometry Layer**:
    *   **Perspective Transform**: A Homography matrix projects 2D image points to road-plane coordinates.
    *   **Metric Conversion**: Pixels are converted to meters using a calibrated `Pixels-Per-Meter` (PPM) scale.
4.  **Fusion & Logic**:
    *   Distances are calculated relative to the "Ego Car" center.
    *   Safety risks are assessed based on longitudinal distance.
5.  **Visualization**:
    *   Overlays lane masks and bounding boxes on the main view.
    *   Updates the dynamic BEV Minimap.

---

## ⚙️ Installation

### Prerequisites
*   **Python 3.8+**
*   **Git**
*   **(Optional)** CUDA-capable GPU (NVIDIA) or Apple Silicon (available via Metal Performance Shaders - MPS) for faster inference.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/mohsinnyz/LaneCollisionADAS.git
    cd LaneCollisionADAS
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models**
    *   YOLOv8n will perform an auto-download on the first run.
    *   Segformer files are cached automatically by the `transformers` library.

---

## 🔧 Configuration & Calibration

Before running the main system, you must calibrate the camera to understand the road geometry.

### 1. Extract Calibration Frame
If you don't have a still image of the road, extract one from your test video:
```bash
python extract_frame.py
```
*This saves `assets/calibration_target.jpg`.*

### 2. Run Calibration Tool
Launch the interactive calibration script:
```bash
python -m src.geometry.camera_calibration
```
**Calibration Instructions:**
1.  A window will open showing the road image.
2.  Click **4 points** to form a rectangle on the road surface:
    *   **Point 1**: Bottom-Left of the lane.
    *   **Point 2**: Top-Left of the lane.
    *   **Point 3**: Top-Right of the lane.
    *   **Point 4**: Bottom-Right of the lane.
3.  *Tip: Imagine measuring a rectangle on the asphalt. The system assumes a standard lane width of 3.7 meters.*
4.  Press **'c'** to Calculate and Save.
5.  Press **'q'** to Quit.

*This generates `configs/homography_matrix.npy` and `configs/pixels_per_meter.npy`.*

---

## 🚀 Usage

To start the ADAS system:

```bash
python main.py
```

### Controls
*   **`q`**: Quit the application.

### Customization
Open `main.py` to adjust:
*   `VIDEO_PATH`: Path to your input video file.
*   `lane_model`: Switch Segformer variants (e.g., b0 to b5) for speed vs. accuracy trade-offs.

---

## 📂 Project Structure

```text
LaneCollisionADAS/
├── assets/                 # Video and image resources
├── configs/                # Calibration matrices (.npy files)
├── models/                 # Cached model weights
├── src/
│   ├── fusion/             # Sensor fusion logic (Future)
│   ├── geometry/           # Camera calibration & Homography
│   ├── perception/         # AI Models (Lane & Object detection)
│   ├── tracking/           # Object tracking (Future)
│   └── utils/              # Helper functions
├── main.py                 # Application Entry Point
├── extract_frame.py        # Utility: Get frame for calibration
├── requirements.txt        # Python dependencies
└── README.md               # Project Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements, bug fixes, or new features.

---

## 📜 License

This project is licensed under the MIT License.
