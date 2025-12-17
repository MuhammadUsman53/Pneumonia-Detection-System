# 🫁 Pneumonia Detection System

This project is a Deep Learning application built with Python, TensorFlow/Keras, and Streamlit to detect Pneumonia from chest X-ray images. It uses a Transfer Learning approach with the MobileNetV2 architecture.

## 📁 Project Structure

```
project/
├── app.py                # Streamlit Application
├── train_model.py        # Model Training Script
├── requirements.txt      # Project Dependencies
├── README.md             # Documentation
└── dataset/              # Dataset Directory (Not included in repo)
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/             # Optional
        ├── NORMAL/
        └── PNEUMONIA/
```

## 🚀 Setup Instructions

### 1. Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
You need a Pneumonia Chest X-Ray dataset. You can download one from Kaggle:
- [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Important:**
1. Download the dataset.
2. Extract the files.
3. Create a folder named `dataset` in this project directory.
4. Move the `train` (and optionally `test`/`val`) folders into `dataset`.
   - Ensure you have `dataset/train/NORMAL` and `dataset/train/PNEUMONIA`.

### 3. Organize Dataset
To enable Bacteria vs Virus detection, we need to sort the dataset.
1. Ensure your `dataset` folder is structured as:
   ```
   dataset/
   └── train/
       ├── NORMAL/
       └── PNEUMONIA/
   ```
2. Run the organization script:
   ```bash
   python organize_dataset.py
   ```
   This will automatically split `PNEUMONIA` into `BACTERIA` and `VIRUS`.

### 4. Train the Model
Run the training script to generate the `pneumonia_model.h5` file:
```bash
python train_model.py
```
*Note: Make sure your `dataset` folder is set up correctly before running this.*

### 4. Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

## 🌐 Deployment (Streamlit Cloud)

1. Push this code to a GitHub repository (ensure `dataset` is ignored via `.gitignore`).
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub account and select your repository.
4. Select `app.py` as the main file.
5. Click **Deploy**!

## 🧪 Model Details
- **Architecture**: MobileNetV2 (Pre-trained on ImageNet) + Custom Dense Head
- **Input Size**: 224x224
- **Classes**: Normal, Pneumonia

## 📸 Screenshots
*(Add your screenshots here after running the app)*

