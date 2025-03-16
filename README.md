# Cephalometric Landmark Detection using U-Net Architecture

This repository provides the source code and sample dataset for the paper:

**"Enhanced Automatic Detection of Cephalometric Landmarks in Pediatric Orthodontics using U-Net Architecture"**

📄 **Citation:**  
If you use this code or dataset, please cite our work as described in the paper.

---

## 🔍 Overview

This project introduces a deep learning-based system based on a customized U-Net model for automatic detection of cephalometric landmarks in pediatric orthodontics. The model is trained on 1200 lateral cephalometric radiographs annotated by expert orthodontists.

The goal is to improve diagnostic accuracy, reduce manual workload, and enhance consistency in orthodontic analysis of growing patients.

---

## 📁 Repository Structure

```
cephalometric-landmark-detection/
├── dice_loss_cephalometric_landmarks.ipynb     # Main Jupyter Notebook
├── new1200/
│   ├── ceph400/                                # Cephalometric X-ray images
│   ├── train_senior.csv                        # Landmark annotations (training set)
│   ├── test1_senior.csv                        # Test set 1
│   └── test2_senior.csv                        # Test set 2
├── initial_model_predictions.csv               # Predicted landmarks (before correction)
├── adjusted_model_predictions.csv              # Predicted landmarks (after correction)
├── New_test.jpg                                # Sample image
└── README.md
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

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/SHAHAB-K93/automated_cephalometry.git
cd automated_cephalometry
```

2. **Prepare the dataset:**
- Place X-ray images in `new1200/ceph400/`
- Annotation format in CSV: `Image_Name, 1_x, 1_y, 2_x ,2_y ...`

3. **Run the code (Jupyter):**
```bash
jupyter notebook dice_loss_cephalometric_landmarks.ipynb
```

4. *(Optional)* Run with Python script (if converted):
```bash
python run_model.py
```

---

## 📑 Code Description

### 🔹 **Data Loading & Preprocessing**
- Load coordinates from CSV
- Generate landmark masks (radius = 4 px)
- Gaussian filtering
- Resize images/masks (256x256)

### 🔹 **U-Net Architecture**
- Encoder: Conv2D, BatchNorm, ReLU, MaxPooling
- Decoder: Conv2DTranspose with skip-connections
- Final layer: 1x1 Conv2D

### 🔹 **Model Training**
- Loss: Dice Loss
- Optimizer: Adam (lr=0.001)
- Epochs: 400 | Batch Size: 16 | Metric: Dice Coefficient

### 🔹 **Post-processing**
- Predicted landmarks from heatmaps (peak points)
- Normalization based on Sella–Nasion (SN_Length) distance
- Correction using average offset per landmark

---

## 📂 Dataset & Code Access

- **Code:** Included in this repository
- **Sample Dataset:** `/new1200/`
- **DOI:** https://doi.org/10.5281/zenodo.15020583

(*Trained weights can be provided upon request.*)

---

## 📌 Notes

- All coordinates normalized relative to SN length
- Post-correction significantly improves prediction accuracy
- Optimized for pediatric orthodontics

---

## ✉ Contact

**Dr. Shahab Kavousinejad**  
Department of Orthodontics, School of Dentistry, Shahid Beheshti University of Medical Sciences  
📧 dr.shahab.k93@gmail.com  
*(On behalf of all co-authors)*
