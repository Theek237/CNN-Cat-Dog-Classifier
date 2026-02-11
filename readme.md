# CNN Cat and Dog Classifier 🐱🐶

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of cats and dogs with high accuracy.

## 🎯 Project Overview

This project implements a binary image classifier that can distinguish between cats and dogs using deep learning. The CNN model is trained to extract hierarchical features from input images and make accurate predictions through multiple convolutional and pooling layers.

## 🏗️ Project Structure

```
CNN-Cat-Dog-Classifier/
├── models/
│   └── catdogclassifier.keras     # Trained model file
├── notebooks/
│   ├── catdog.ipynb              # Data preprocessing and exploration
│   ├── gettingpredictions.ipynb  # Model inference and predictions
│   ├── modeltrain.ipynb          # Model training notebook
│   └── res.ipynb                 # Results and evaluation
├── images/                       # Sample prediction images
└── readme.md                     # Project documentation
```

## 🚀 Features

- **High Accuracy**: Achieves excellent classification performance
- **Binary Classification**: Distinguishes between cats and dogs
- **CNN Architecture**: Uses convolutional layers for feature extraction
- **Ready-to-Use Model**: Pre-trained model available in Keras format
- **Interactive Notebooks**: Jupyter notebooks for training and inference

## 🔧 Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📋 Requirements

```bash
tensorflow>=2.0
keras
opencv-python
numpy
matplotlib
jupyter
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CNN-Cat-Dog-Classifier.git
   cd CNN-Cat-Dog-Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (if training from scratch)
   ```bash
   # Follow instructions in catdog.ipynb for dataset download
   ```

## 📚 Usage

### Making Predictions

1. **Load the pre-trained model**
   ```python
   from keras.models import load_model
   model = load_model('models/catdogclassifier.keras')
   ```

2. **Prepare your image**
   ```python
   import cv2
   import numpy as np
   
   # Load and preprocess image
   img = cv2.imread('path/to/your/image.jpg')
   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   img = cv2.resize(img, (100, 100))
   img = img/255.0
   img = img.reshape(1, 100, 100, 1)
   ```

3. **Get prediction**
   ```python
   prob = float(model.predict(img))
   
   if prob >= 0.5:
       print(f"Cat 🐱 ({prob*100:.2f}%)")
   else:
       print(f"Dog 🐶 ({(1-prob)*100:.2f}%)")
   ```

### Training Your Own Model

1. Open and run [notebooks/catdog.ipynb](notebooks/catdog.ipynb) for data preparation
2. Use [notebooks/modeltrain.ipynb](notebooks/modeltrain.ipynb) to train the model
3. Evaluate results with [notebooks/res.ipynb](notebooks/res.ipynb)

## 🧠 Neural Network Architecture

The CNN model implements a sophisticated deep learning architecture:

### Input Layer
- **Input Shape**: 100x100x1 (grayscale images)
- **Preprocessing**: Images normalized to [0,1] range

### Convolutional Blocks
- **Multiple Conv2D layers** with ReLU activation functions
- **Feature maps** of varying sizes (32, 64, 128 filters)
- **Kernel sizes** typically 3x3 for optimal feature extraction
- **Padding**: 'same' to preserve spatial dimensions

### Pooling Layers
- **MaxPooling2D layers** (2x2 pool size) for spatial dimension reduction
- **Feature map compression** while retaining important information
- **Translation invariance** for robust object detection

### Regularization
- **Dropout layers** (typically 0.25-0.5) to prevent overfitting
- **Batch normalization** for stable training (if implemented)

### Classification Head
- **Flatten layer** to convert 2D feature maps to 1D vector
- **Dense layers** with ReLU activation for feature combination
- **Final dense layer** with sigmoid activation for binary classification
- **Output**: Single probability value (0-1) where >0.5 indicates cat, <0.5 indicates dog

### Training Configuration
- **Loss Function**: Binary crossentropy
- **Optimizer**: Adam (typically with learning rate scheduling)
- **Metrics**: Accuracy, precision, recall

## 📊 Model Characteristics

- **Binary Classification**: Outputs probability scores for cat vs dog
- **Grayscale Processing**: Efficient single-channel input
- **Compact Size**: Optimized for deployment
- **Real-time Inference**: Fast prediction capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset from Kaggle's Cat and Dog Classification dataset
- TensorFlow/Keras community for excellent documentation
- OpenCV for image processing capabilities

## 📞 Contact

Feel free to reach out if you have any questions or suggestions!

---

⭐ **Star this repository if you found it helpful!** ⭐