# Neural Style Transfer

An implementation of neural style transfer using Adaptive Instance Normalization (AdaIN) with TensorFlow and PyTorch.

## Overview

This project implements a neural style transfer model that applies the style of one image to the content of another image. The implementation uses an encoder-decoder architecture with AdaIN (Adaptive Instance Normalization) to achieve real-time style transfer while preserving the content structure.

## Features

- **AdaIN-based Style Transfer**: Uses Adaptive Instance Normalization to match feature statistics
- **VGG Loss Network**: Perceptual loss function based on VGG features for content and style preservation
- **Encoder-Decoder Architecture**: Efficient feature extraction and image reconstruction
- **Multi-framework Support**: Built with both TensorFlow/Keras and PyTorch
- **Training and Evaluation**: Comprehensive training pipeline with loss tracking and visualization

## Technical Details

### Key Components

- **Encoder**: Extracts feature representations from images
- **Decoder**: Reconstructs stylized images from encoded features
- **Loss Network**: VGG-based perceptual loss for measuring content and style similarity
- **AdaIN**: Matches the mean and variance of content features to style features

### Loss Functions

- **Content Loss**: MSE between encoded content features and decoder output
- **Style Loss**: MSE between mean/variance statistics of style and reconstructed features
- **Total Loss**: Combined weighted loss for optimization

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- TensorFlow 2.13.0
- PyTorch 2.0.1 + TorchVision 0.15.2
- NumPy 1.24.3
- Scikit-learn 1.3.0
- Matplotlib 3.7.2
- Pandas 2.0.3
- Seaborn 0.12.2
- TQDM 4.66.0
- LMDB 1.4.1
- Ninja 1.11.1

## Language Composition

- Python: 91%
- CUDA: 7.6%
- C++: 1.4%

## Project Structure

```
.
├── models/           # Model architecture and training
│   ├── trainer.py    # Training pipeline and loss computation
│   ├── network.py    # Encoder and decoder networks
│   ├── loss.py       # VGG loss network
│   └── data_loader.py # Data loading utilities
├── localtoon/        # LocalToon style transfer variant with StyleGAN2
├── notebooks/        # Jupyter notebooks for exploration
├── data/             # Training and test data
├── explore.ipynb     # Data exploration notebook
├── create_gif.py     # Utility for creating animated GIFs
└── requirements.txt  # Python dependencies
```

## Usage

### Training

```python
from models.trainer import train, get_model

# Train the model
model = get_model()
history, trained_model = train(model=model)
```

### Style Transfer Inference

```python
from models.trainer import NeuralStyleTransfer
from tensorflow.keras.utils import array_to_img

# Load trained model
model = tf.keras.models.load_model('saved_models/art_style/art_style.keras')

# Apply style transfer
style_image = load_image('path/to/style.jpg')
content_image = load_image('path/to/content.jpg')
stylized_output = model(style_image, content_image)
```

## Training Details

- **Epochs**: 30
- **Optimizer**: Adam (learning rate: 1e-5)
- **Loss Function**: Mean Squared Error
- **Style Weight**: 4.0
- **Validation**: Includes test set evaluation

## Monitoring

The training process includes a `TrainMonitor` callback that:
- Visualizes style, content, and stylized output at each epoch
- Saves generated images for analysis
- Tracks training progress

## Output

The model generates stylized images that:
- Preserve the structure and content of the input image
- Adopt the visual style (colors, textures, patterns) of the style reference
- Maintain high perceptual quality through VGG-based loss functions

## References

The implementation is based on Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN) architecture.
