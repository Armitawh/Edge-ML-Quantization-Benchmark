# Edge ML Quantization Benchmark (PyTorch)

## 📌 Overview
This project demonstrates **hardware-aware optimization of deep learning models** for edge and embedded deployment using **post-training INT8 quantization** in PyTorch.

A convolutional neural network (CNN) is trained on the MNIST dataset and then optimized using **dynamic quantization** to reduce model size and analyze inference performance trade-offs. The project includes **benchmarking and visualization** of accuracy, latency, and memory footprint.

This work is designed to reflect real-world constraints faced in **edge AI, embedded ML, and ML compiler workflows**.

---

## 🎯 Objectives
- Train a baseline CNN model for image classification
- Apply **post-training INT8 quantization**
- Benchmark:
  - Model size
  - Inference latency
  - Classification accuracy
- Visualize optimization trade-offs
- Demonstrate **hardware-aware ML deployment skills**

---

## 🧠 Key Concepts
- Convolutional Neural Networks (CNNs)
- Post-training Dynamic Quantization
- INT8 vs FP32 inference
- Model size vs latency trade-offs
- Edge ML performance benchmarking

---

## 🏗 Model Architecture
- **Input:** 28×28 grayscale image
- **Conv Layers:** Feature extraction
- **Max Pooling:** Spatial downsampling
- **Fully Connected Layers:** Classification
- **Output:** 10 digit classes (0–9)

---

## 🛠 Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 📊 Results Overview

The CNN model was trained for 3 epochs on the MNIST dataset and optimized using post-training INT8 dynamic quantization. Training converged smoothly with decreasing loss values, confirming stable learning behavior.

### 🔹 Training Loss
- Epoch 1: 0.1716  
- Epoch 2: 0.0566  
- Epoch 3: 0.0397  

This demonstrates effective feature learning and fast convergence.

---

### 🔹 Performance Comparison

| Model | Size (KB) | Accuracy (%) | Inference Time (s) |
|------|----------|--------------|--------------------|
| FP32 (Original) | **4690.61** | **98.97** | **5.04** |
| INT8 (Quantized) | **1232.20** | **98.98** | **5.00** |

---

### 🔹 Key Observations
- **Model size reduced by ~74%** using INT8 quantization.
- **Accuracy was preserved**, with no measurable degradation after quantization.
- **Inference latency slightly increased** on a CPU-only environment, highlighting the importance of hardware-specific INT8 acceleration (e.g., ARM NEON, DSP, NPU).

This result reflects a real-world edge AI trade-off: quantization significantly reduces memory footprint while latency gains depend on the underlying hardware and runtime backend.

