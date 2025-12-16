# COVID-19 Detection Using Chest X-ray Images

This repository contains the Jupyter notebook, project documentation, and output screenshots for the project **“COVID-19 Detection Using Chest X-ray Images with Deep Learning”**.

---

# Project Overview

This project implements a deep learning–based medical image classification system to detect COVID-19 from chest X-ray images using Convolutional Neural Networks (CNN).

The system is developed using:

- Python  
- TensorFlow / Keras  
- Convolutional Neural Networks (CNN)  
- Jupyter Notebook / Google Colab  

The model aims to support early diagnosis by automatically classifying chest X-ray images into COVID-19 and Non-COVID categories.

---

# System Architecture

## Dataset
- Chest X-ray image dataset  
- Images labeled as COVID-19 and Non-COVID  
- Data preprocessing includes resizing, normalization, and augmentation  

---

## Model Architecture

The deep learning model consists of:

- Convolutional layers for feature extraction  
- Max-pooling layers for dimensionality reduction  
- Fully connected (Dense) layers for classification  
- Activation functions such as ReLU and Softmax/Sigmoid  

The model is trained using supervised learning and evaluated using accuracy and loss metrics.

---

# Implementation Details

The complete implementation is performed in a Jupyter Notebook:

- Loading and preprocessing the dataset  
- Building the CNN architecture  
- Training and validating the model  
- Evaluating performance  
- Visualizing accuracy, loss, and predictions  

Notebook file:
- `COVID19_DL.ipynb`

---

# Repository Structure

```
Covid-19-detection-using-chest-X-ray-Images/
│
├── notebooks/
│ └── COVID19_DL.ipynb
│
├── report/
│ ├── E0322005_REPORT.pdf
│ └── Deep_Learning_COVID19_Report.pdf
│
├── results/
│ └── (Accuracy graphs, loss graphs, output screenshots)
│
└── README.md
```


---

# How to Run the Project

## Step 1: Open the Notebook
- Navigate to the `notebooks/` folder  
- Open `COVID19_DL.ipynb` in Jupyter Notebook or Google Colab  



## Step 2: Install Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```


## Step 3: Execute the Notesbook

-Run all cells sequentially
-The notebook will:
	-Load the dataset
	-Train the CNN model
	-Evaluate model performance
	-Display output graphs and predictions

---

# Results 

-Model accuracy and loss graphs are generated
-Training and validation results are visualized
-Prediction outputs are stored in the results/ folder

---

# Documentation

-Complete project report explaining methodology and results
-Academic documentation and analysis
-All documentation files are available in the report/ folder.

---

# Conclusion

This project demonstrates the effective use of deep learning techniques in healthcare applications by detecting COVID-19 from chest X-ray images. The system highlights the potential of CNN models in assisting medical professionals with automated diagnosis.

---

## Author

**Keerthana S**  
B.Tech – CSE (Artificial Intelligence and Data Analytics)





