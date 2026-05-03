# Emotion Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A robust **Machine Learning/Deep Learning** system that identifies human emotions from facial expressions in real-time. This project uses YOLOv8 for face detection and a custom CNN for emotion classification.

## Table of Contents

- [Features](#features)
- [Supported Emotions](#supported-emotions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Performance](#performance)
- [License](#license)

## Features

✨ **Real-time emotion detection** from webcam feed or video files  
✨ **Face detection** using YOLOv8 Nano model  
✨ **Multi-emotion classification** with confidence scores  
✨ **GPU acceleration support** (CUDA-enabled)  
✨ **Easy-to-use API** for integration into other projects  
✨ **Modular design** with separate detector and classifier components  

## Supported Emotions

The system classifies facial expressions into 7 distinct emotions:

1. 😠 **Angry**
2. 😒 **Disgust**
3. 😨 **Fear**
4. 😊 **Happy**
5. 😢 **Sad**
6. 😮 **Surprise**
7. 😐 **Neutral**

## Project Structure

```
Emotion-Detection/
├── main.py                 # Main demo script for real-time detection
├── train.py               # Model training script
├── detector.py            # Face detection module
├── classifier.py          # Emotion classification module
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── models/               # Pre-trained models (to be downloaded)
    ├── yolov8n-face.pt   # YOLOv8 face detector
    └── emotion_model.pth # Trained emotion classifier
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster processing

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ritti902/Emotion-Detection.git
cd Emotion-Detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

You need to download the pre-trained models:

```bash
mkdir -p models
# Download YOLOv8 face detection model
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-face.pt -O models/yolov8n-face.pt

# Download pre-trained emotion model (or train your own - see Training section)
# Download from your preferred location or train from scratch
```

Alternatively, the first run will auto-download the YOLOv8 model if available.

## Usage

### Quick Start: Real-time Emotion Detection

Run the demo with your webcam:

```bash
python main.py
```

**Controls:**
- Press `Q` or `ESC` to exit the application

### Advanced Usage

#### Use specific camera device:

```bash
python main.py --camera 1  # Use camera index 1 instead of default
```

#### Process video file:

Create a modified version of `main.py` or use the detector/classifier modules:

```python
from detector import FaceDetector
from classifier import EmotionClassifier
import cv2

detector = FaceDetector("models/yolov8n-face.pt")
clf = EmotionClassifier(model_path="models/emotion_model.pth")

cap = cv2.VideoCapture("path/to/video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect(frame)
    for (x1, y1, x2, y2) in faces:
        face = frame[y1:y2, x1:x2]
        label, prob = clf.predict(face)
        # Process results...
    
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Process single image:

```python
from detector import FaceDetector
from classifier import EmotionClassifier
import cv2

detector = FaceDetector("models/yolov8n-face.pt")
clf = EmotionClassifier(model_path="models/emotion_model.pth")

frame = cv2.imread("path/to/image.jpg")
faces = detector.detect(frame)

for (x1, y1, x2, y2) in faces:
    face = frame[y1:y2, x1:x2]
    label, prob = clf.predict(face)
    print(f"Emotion: {label}, Confidence: {prob:.2%}")
```

## Training

### Prerequisites for Training

1. **FER2013 Dataset** - Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Place the `fer2013.csv` file in `datasets/` directory:

```bash
mkdir -p datasets
# Place fer2013.csv in this directory
```

### Train the Emotion Model

```bash
python train.py
```

This will:
- Load the FER2013 dataset (28,709 training samples)
- Train a CNN model for 100 epochs
- Save the trained model to `models/emotion_model.pth`
- Perform 10% validation split

### Customize Training Parameters

Edit `train.py` to modify:
- **Batch size**: Change `batch_size` in DataLoader (line 63)
- **Learning rate**: Modify `lr` in optimizer (line 70)
- **Number of epochs**: Adjust range in training loop (line 73)
- **Model architecture**: Modify `EmotionCNN` class (lines 30-47)

## Model Architecture

### Face Detection: YOLOv8 Nano

- **Model**: YOLOv8n-face (lightweight nano variant)
- **Input**: RGB images at variable resolution
- **Output**: Bounding boxes for detected faces
- **Performance**: ~40-60 FPS on CPU, 100+ FPS on GPU

### Emotion Classification: Custom CNN

```
Input (1, 48, 48)
    ↓
Conv2d(1→32, 3x3) + ReLU + MaxPool(2x2)
    ↓
Conv2d(32→64, 3x3) + ReLU + MaxPool(2x2)
    ↓
Flatten
    ↓
Linear(64*12*12→128) + ReLU + Dropout(0.5)
    ↓
Linear(128→7) [Output: 7 emotion classes]
```

**Architecture Details:**
- **Total Parameters**: ~67,000
- **Input Size**: 48×48 grayscale images
- **Output**: 7 emotion classes with softmax probabilities
- **Training Method**: Cross-entropy loss with Adam optimizer

## Requirements

```
torch              # Deep learning framework
torchvision        # Computer vision utilities
ultralytics        # YOLOv8 implementation
opencv-python      # Image processing
numpy              # Numerical computing
scikit-learn       # Machine learning utilities
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

### FER2013 (Facial Expression Recognition 2013)

- **Size**: 35,887 images
- **Image Format**: 48×48 grayscale
- **Emotions**: 7 classes (angry, disgust, fear, happy, sad, surprise, neutral)
- **Train/Test Split**: 28,709 training, 7,178 public test
- **Imbalance**: Some emotions underrepresented (e.g., disgust)
- **Download**: [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Performance

| Metric | Value |
|--------|-------|
| **Inference Speed (CPU)** | ~40-60 FPS |
| **Inference Speed (GPU)** | ~100+ FPS |
| **Model Size** | ~2.5 MB (detector + classifier) |
| **Memory Usage** | ~200-400 MB |

**Note:** Actual performance depends on:
- Hardware specifications
- Image resolution
- Number of faces per frame
- GPU availability

## Troubleshooting

### "ModuleNotFoundError" when running main.py

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Camera not opening

**Solution**: 
- Check camera permissions
- Try different camera index: `python main.py --camera 1`
- Verify no other application is using the camera

### Model file not found

**Solution**:
- Create `models/` directory: `mkdir models`
- Download pre-trained models to this directory
- Or train your own model: `python train.py`

### Low accuracy on custom images

**Solution**:
- Ensure face is clearly visible and well-lit
- Try cropping the face to focus on the center
- Retrain model on custom dataset for better domain adaptation

## Future Improvements

- [ ] Add support for multiple face recognition
- [ ] Implement head pose estimation
- [ ] Add facial action unit detection
- [ ] Deploy as web service (Flask/FastAPI)
- [ ] Create mobile app version
- [ ] Add intensity/confidence calibration
- [ ] Support for masked faces

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit with clear messages (`git commit -m 'Add feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **FER2013 Dataset** by [Pierre Luc Carrier and Aaron Courville](https://www.kaggle.com/datasets/msambare/fer2013)
- **PyTorch** by Meta AI Research

## Contact & Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Open an issue](https://github.com/Ritti902/Emotion-Detection/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/Ritti902/Emotion-Detection/discussions)

---

**Last Updated**: May 2026  
**Maintainer**: [Ritti902](https://github.com/Ritti902)
