# 🐶🐱 Image Captioning and Segmentation

## 📌 Project Overview  
This project integrates **image segmentation** and **image captioning** using the Oxford-IIIT Pet Dataset.  

- **Segmentation**: Uses **U-Net** to separate pets (cats/dogs) from the background.  
- **Captioning**: Uses **InceptionV3 + LSTM** to generate natural language captions.  
- **Integration**: Final pipeline combines segmentation masks with generated captions for each image.  

---

## 📂 Project Structure  
- ├── captions
- ├── features
- ├── images
- ├── models
- ├── processed
- ├── final_pipeline.py
- ├── Image Captioning and Segmentation Report.pdf
- ├── Image captioning_20250417_182046_0000 (1).pdf
- ├── requirements.txt
- ├── train_captioning.py
- ├── train_segmentation.py
- ├── week3_4_data_preparation.ipynb
- ├── week5_6_model_training.ipynb
- ├── week7_8_segmentation.ipynb
- ├── week9-integration.ipynb
- ├── week10_segmentation_evaluation_metrics.ipynb
- ├── week11_visualization.ipynb
- ├── wek12_final_pipeline.ipynb
- └── README.md

---

## ⚙️ Installation & Setup  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install dependencies
  ```
   pip install -r requirements.txt
  ```

## 🚀 Usage  

### Train Segmentation Model  
```
python train_segmentation.py
```
### Train Captioning Model
```
python train_captioning.py
```
### Run Final Integrated Pipeline
```
python final_pipeline.py
```

## 📊 Results

- **Segmentation Performance**: IoU ≈ 0.70, Dice ≈ 0.82

- **Captioning**: Generates meaningful captions like:

  - "a brown dog running in grass"

  - "a white cat sitting on a couch"

Final output visualization includes:

- Original Image

- Ground Truth Mask

- Predicted Mask

- Generated Caption

## 🌍 Applications

- **Healthcare** – Medical image analysis + descriptive reporting.

- **Autonomous Vehicles** – Object detection + scene understanding.

- **Accessibility** – Helping visually impaired users understand images.

## 📌 Future Work

- Fine-tune captioning on larger datasets.

- Explore advanced segmentation models (DeepLabV3+, Mask R-CNN).

- Deploy as a web app using Streamlit/Flask.

