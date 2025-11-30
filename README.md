#  Human Action Recognition Using CNN (CSV-Mapped Image Labels)

This project implements a **Convolutional Neural Network (CNN)** to classify **human actions from images**.  
Unlike standard datasets, the images are not arranged in class-wise folders. Instead:

- All images are stored inside `train/` and `test/` folders  
- Training labels are stored in `Training_set.csv`  
- Test set contains only filenames (like Kaggle competitions)

This requires a **custom pipeline** for loading, preprocessing, and training.

---

##  Project Overview

The goal is to build a deep learning model capable of identifying various human actions such as:

- Walking  
- Sitting  
- Sleeping  
- Using Laptop  
- Hugging  
- Running  
- And other daily activities  

The project uses a CNN architecture trained on manually loaded images and CSV-based labels.

---

##  Dataset Structure

Human Action Recognition/
│
├── train/
│ └── all training images
│
├── test/
│ └── all test images
│
├── Training_set.csv → filename + label
└── Testing_set.csv → filename only

---

##  Pipeline Steps

### **1. Load CSV files**  
Reads filenames and labels from the CSV.

### **2. Load & preprocess images**  
- Resize to (64×64)  
- Convert to arrays  
- Normalize pixel values  

### **3. Encode labels**  
- LabelEncoder → converts text labels to integers  
- One-hot encoding → converts integers to vectors  

### **4. Train–validation split**  
Using `train_test_split` from sklearn.

### **5. Build CNN model**  
Includes:
- Convolution layers  
- MaxPooling  
- Dense layers  
- Dropout for regularization  
- Softmax output  

### **6. Train the model**  
The CNN learns patterns for action identification.

### **7. Plot training performance**  
Accuracy & loss curves for analysis.

### **8. Predict actions on test set**  
The model generates predicted action labels for unseen images.

### **9. Save model & label encoder**  
Saved for future deployment or inference.

---

##  CNN Architecture

Input (64x64x3)
↓
Conv2D (32 filters)
↓
MaxPooling2D
↓
Conv2D (64 filters)
↓
MaxPooling2D
↓
Conv2D (128 filters)
↓
MaxPooling2D
↓
Flatten
↓
Dense (128 units)
↓
Dropout (0.5)
↓
Dense (num_classes, Softmax)

---

## Training Performance

- Training accuracy increased with epochs  
- Validation accuracy stabilized around ~30–40%  
- The model successfully predicts actions on test images  

---

##  Example: Prediction Code

```python
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_classes)
print(predicted_labels[:10])
model.save("har_cnn_model.h5")
