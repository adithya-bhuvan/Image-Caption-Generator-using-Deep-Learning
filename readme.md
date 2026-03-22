# Image Caption Generator using Deep Learning

## Description
This project generates captions for images using a combination of CNN and LSTM models.

## Model Architecture
- CNN (feature extraction)
- LSTM (caption generation)
- Encoder-Decoder framework

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV, Matplotlib

## Files
- model.py → Model architecture
- train.py → Training script
- predict.py → Caption generation

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train model:
   python train.py

3. Generate caption:
   python predict.py

## Dataset
- Flickr8k / MSCOCO

## Output
Generates captions like:
"A dog is running in the park"