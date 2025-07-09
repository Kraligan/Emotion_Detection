# Emotion Detection

![Webcam Demo](assets/emotion_demo.gif)

---

## Objective

Build a lightweight facial emotion recognition system using computer vision. Two main strategies were considered:

- Train a CNN directly on images (e.g. FER2013, AffectNet)
- Extract facial landmarks with MediaPipe and classify them using a simpler model (MLP)

We chose the second approach to keep the model small and efficient, suitable for real-time webcam inference using only 3D facial landmarks (x, y, z).

---

## Project Steps

### Step 1 – First Attempt with FER2013

- Dataset: `msambare/fer2013` from KaggleHub
- Extracted landmarks from 48x48 images using MediaPipe
- Issues:
  - Image resolution too low for reliable landmark extraction
  - MLP model reached only ~15% accuracy
- **Conclusion:** this dataset is unsuitable for a landmark-based pipeline

### Step 2 – Switch to AffectNet

- Dataset: `mstjebashazida/affectnet`, with higher-quality, varied images
- Organized into Train/ and Test/ folders by emotion class
- Landmarks extracted using MediaPipe (468 x, y, z points)

### Step 3 – Cleanup and Dataset Balancing

- Goal: create a balanced dataset with a fixed number of samples per emotion class
- Actions:
  - Merged Train/ and Test/
  - Reduced to **4 emotion classes**: `happy`, `sad`, `fear`, `neutral`
  - Kept up to 2500 images per class
  - Result saved in `dataset_Affectnet_balanced.txt` with 1404 features + label

### Step 4 – Preprocessing Pipeline

- Data loaded using NumPy
- Landmark values standardized with `StandardScaler` (fit on train set only)
- Data split using stratified train/val/test (10% test, 20% validation)

### Step 5 – MLP Model Architecture

Final MLP used for AffectNet training:

```
Dense(1024) → BatchNorm → Dropout(0.5)
Dense(512)  → BatchNorm → Dropout(0.5)
Dense(256)  → BatchNorm → Dropout(0.4)
Dense(128)  → Dropout(0.3)
Dense(4)    → Softmax
```

- Optimizer: Adam with learning rate = 1e-5
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`
- Class weights computed using `class_weight='balanced'`

### Step 6 – Evaluation

- Final test accuracy ~54%
- Validation accuracy plateaued around ~55–56%
- Confusion matrix revealed the model performed better on `happy` and `neutral`, but struggled with `fear`

### Step 7 – Webcam Inference

- Integrated real-time webcam inference with OpenCV
- Used MediaPipe to extract landmarks frame-by-frame
- Normalized landmarks using the same `StandardScaler`
- Predicted emotion displayed with confidence on live feed
- Model struggled to recognize some emotions like `fear` consistently on real webcam feed

### Step 8 – Fine-Tuning on Personal Dataset

- Created a small personal dataset using webcam, annotated manually (4 emotions)
- Two strategies tested:
  - Training from scratch on personal dataset → overfitting due to limited data
  - **Fine-tuning** on top of pre-trained AffectNet MLP → **significant improvement**
- Strategy: freeze early layers, re-train last few layers on personal samples
- Result: model now better recognizes personal expressions

---

## Results

- Final validation accuracy on personal dataset: **~76%**
- Webcam inference greatly improved for personal expressions (except `fear`, still less stable)

---

## Limitations and Future Improvements

- Using only 3D landmarks limits the expressiveness captured compared to full image input (e.g. CNNs)
- MediaPipe landmark extraction may fail on low-light or occluded faces
- Facial expressions like `fear` can be subtle and may require more contextual features

### Next Steps:

- Explore CNN-based architectures trained on full images (e.g. MobileNet or EfficientNet)
- Combine landmarks and CNN features in a hybrid model
- Add more data (synthetic or real) to better handle rare expressions
- Convert model to TFLite or ONNX for lightweight deployment

---

## How to Use

### Requirements

You can install everything with:

```bash
pip install -r requirements.txt
```

### Run the Webcam Demo

Make sure your webcam is enabled. Then run:

```bash
python scr/main.py
```

### Train on AffectNet

```bash
python scr/train_model.py
```

### Fine-tune on Your Own Dataset

You must first generate your own labeled landmark dataset via webcam using **record_dataset.py**, then run:

```bash
python scr/train_personal.py
```

---

## Challenges & Solutions

| Challenge                                   | Solution                                               |
|--------------------------------------------|--------------------------------------------------------|
| FER2013 images too low-quality              | Switched to AffectNet dataset                          |
| Landmarks model underperforming on webcam  | Fine-tuned on custom labeled dataset                  |
| Model overfitting on small dataset         | Used transfer learning and froze early layers         |
| Poor recognition of `fear` emotion         | Highlighted as limitation, suggested CNN improvements |

