
# Cephalometric Landmark Detection using U-Net Architecture

This repository contains the source code and a sample dataset for the paper:

  "Enhanced Automatic Detection of Cephalometric Landmarks in Pediatric Orthodontics using U-Net Architecture"  

📄   Citation:    
If you use this code or dataset, please cite our work as described in the paper.

---

## 🔍 Overview

This project presents a deep learning-based system for the automatic detection of cephalometric landmarks in growing patients using a U-Net architecture. The model is trained and evaluated on 1200 lateral cephalometric radiographs annotated by expert orthodontists.

The proposed method improves diagnostic accuracy, reduces manual workload, and enhances consistency in pediatric orthodontics.

---

## 📁 Repository Structure

```
📂 cephalometric-landmark-detection/
├── dice-loss-cephalometric-landmarks-detection (final2).ipynb   # Main Jupyter Notebook
├── new1200/
│   ├── ceph400/                       # Folder for cephalometric X-ray images
│   ├── train_senior.csv              # CSV file with training landmark coordinates
│   ├── test1_senior.csv              # CSV file with first test set
│   └── test2_senior.csv              # CSV file with second test set
├── landmark_correction_data.csv      # CSV file for post-processing correction adjustments
├── New_test.jpg                      # A new sample image for testing
└── README.md                         # Instructions and guidelines

```

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 1. Clone the repository:
```bash
git clone https://github.com/SHAHAB-K93/automated_cephalometry.git
cd automated_cephalometry
```

### 2. Prepare dataset:
- Place your X-ray images in `new1200/ceph400/ceph400/`
- Provide landmark annotations in `new1200/train_senior.csv` with headers: `Image_Name, 1_x, 1_y, 2_x ,2_y ...`
- A clearer example might be: "Image_Name, 1_x, 1_y, 2_x, 2_y, ... where 1_x and 1_y represent the x and y coordinates of the first landmark (e.g., Sella)."

### 3. Run the code:
Open the Jupyter Notebook:
```bash
jupyter notebook dice-loss-cephalometric-landmarks-detection (final2).ipynb
```

---

## 📑 Code Breakdown

### 🔹 Data Loading & Preprocessing
- Load landmark coordinates from CSV
- Generate binary masks for each landmark (circle radius = 4 pixels)
- Apply Gaussian filter for smoothing masks
- Resize input images and masks to 256x256 pixels

### 🔹 U-Net Model Architecture
- Contracting Path (Encoder): series of Conv2D, BatchNorm, ReLU, MaxPooling, Dropout
- Expanding Path (Decoder): Conv2DTranspose layers with skip connections from encoder
- Final layer: 1x1 Conv2D for output mask prediction

### 🔹 Training
-   Loss function:   Dice Loss
-   Optimizer:   Adam (lr=0.001)
-   Epochs:   400,   Batch size:   16
-   Metrics:   Dice Coefficient, MAE, RMSE, Precision, Accuracy

### 🔹 Post-processing & Correction
- Landmarks predicted from heatmaps (maximum value per channel)
- Standardization using distance between Sella (S) and Nasion (Na) landmarks for each image.
- Post-prediction corrections using an average offset derived from the training data to improve accuracy.

### 🔹 Evaluation
- Euclidean distance error calculation
- Convert pixel distances to millimeters based on the cephalometric ruler scale in landmark_correction_data.csv.
- Compare performance before and after adjustment.

---

## 📊 Results Summary

| Metric     | Before Adjustment | After Adjustment |
|------------|-------------------|------------------|
| Mean Error | 1.28 ± 1.44 mm    | 0.45 ± 0.50 mm   |
| Accuracy   | Up to 0.99        | Up to 0.99       |
| Processing Time | 0.49 sec/image | 0.49 sec/image   |

---

## 📂 Dataset & Code Access

-   Code:   Included in this repository
-   Sample Dataset:   `new1200/` directory

Note: Full dataset not public due to privacy restrictions. A sample dataset is included for demonstration purposes.

---

## 📌 Notes

- All coordinates in landmark_correction_data.csv are normalized relative to S-N line (Sella-Nasion) of each patient.
- Post-prediction corrections improve precision
- Designed specifically for pediatric orthodontics and growing patients

---

## ✉ Contact

  Dr. Shahab Kavousinejad    
Department of Orthodontics, School of Dentistry, Shahid Beheshti University of Medical Sciences, Tehran, Iran  
📧 dr.shahab.k93@gmail.com

(on behalf of all authors)
