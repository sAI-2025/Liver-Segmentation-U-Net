# Liver Segmentation using U-Net

## Project Description
This project implements an improved liver segmentation model using the **U-Net** architecture. The model achieves **89.6% Intersection over Union (IOU)** and **87.4% Dice Coefficient**, demonstrating its effectiveness in segmenting liver images. To enhance model performance, multiple loss functions were experimented with, including:
- Dice Loss
- Binary Cross Entropy (BCE)
- Hybrid Loss
- K-Loss

## Project Structure
```
.
├── Unet [liver].ipynb      # Jupyter notebook with the U-Net implementation
├── Results/                # Model results, images, predicted masks, and metrics
└── Dataset/                # Liver segmentation dataset
    ├── nil files/          # Input images (X)
    └── labels/             # Ground truth masks (Y)
```

## About the Dataset
The dataset consists of liver images where each pixel is labeled, allowing precise segmentation of liver structures. The dataset is in **NIfTI (NIBIL) format**, which is commonly used for medical imaging. It is structured to allow the training of computer vision models to automatically segment liver regions in medical images.

## Implementation Steps
### Prerequisites
Ensure that **Python 3.10** is installed.

1. **Check if pip is installed:**
   ```bash
   pip --version
   ```
   If pip is outdated, upgrade it:
   ```bash
   python3.10 -m pip install --upgrade pip
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup:**
   - Ensure your dataset is structured as mentioned in the project structure.
   - Run the Jupyter notebook.
   - If the dataset is missing, it will be automatically downloaded.

## U-Net Architecture Overview
The **U-Net** architecture is a **fully convolutional neural network** (CNN) designed for biomedical image segmentation. It consists of two main components:

### 1. Contracting Path (Encoder)
- Uses convolution and pooling layers to extract key image features while reducing spatial dimensions.
- Captures fine details necessary for precise segmentation.

### 2. Expanding Path (Decoder)
- Uses convolution and up-convolution (upsampling) to reconstruct the image from feature maps.
- Combines learned features to generate a precise segmentation mask.

### Key Features of U-Net:
- **Skip connections:** Preserve fine-grained spatial information by combining encoder and decoder outputs.
- **Fully convolutional:** No dense layers, allowing for input images of varying sizes.
- **Efficient processing:** Achieves high segmentation accuracy with relatively fewer parameters.

### Performance Metrics:
- **Accuracy:** U-Net provides segmentation accuracy on par with other advanced models.
- **Efficiency:** The optimized model significantly reduces parameters and computational cost while maintaining segmentation quality.

## Loss Functions Used
To improve model performance, multiple loss functions were evaluated:

- **Dice Loss:** Measures the overlap between predicted and ground truth masks, improving segmentation performance.
- **Binary Cross Entropy (BCE):** Standard loss for binary classification tasks.
- **Hybrid Loss:** A combination of Dice Loss and BCE for improved learning.
- **K-Loss:** A customized loss function tailored for medical image segmentation.

## Conclusion
This project successfully implements liver segmentation using **U-Net**, achieving high accuracy and efficiency. The model can be extended to other medical imaging applications, making it a valuable tool for biomedical research.

---
🚀 **Future Improvements:**
- Experiment with **attention-based U-Net** for better localization.
- Fine-tune hyperparameters for **higher segmentation accuracy**.
- Extend to multi-organ segmentation tasks.

📌 **Author:** Sai Krishna Chowdary Chundru

