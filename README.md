# ASL Detection Using CNN

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

  [Kindly visit the original github repository `https://github.com/SHAIMA-HAQUE/ASL_SIGNS` to check out the contributors of this project]

## Introduction
This project focuses on the detection and classification of American Sign Language (ASL) gestures using a Convolutional Neural Network (CNN).
The goal is to build a model that can accurately recognize ASL signs from images, facilitating communication for individuals with hearing or speech impairments.

## Project Workflow
![image](https://github.com/jojangandha-saha/ASL_SIGNS/assets/86916920/1466acaf-90c4-4247-8369-04c3aea6e788)


## Dataset
The dataset used for training and testing the CNN model comprises images of various ASL gestures. Each image is labeled with the corresponding ASL letter or sign. 
Dataset is created by each of the team mates of this academic project.

Ensure that you have the necessary permissions and licenses to use the dataset in your project. Each temmate had captured 5 asl signs and add respective labels. Thus, creating a dataset of 30 
images of isl signs( English phrases). Kindly find the dataset in the - Dataset.zip folder.


## Model Architecture
The CNN model is designed to extract features from the input images and classify them into the appropriate ASL sign categories. The architecture typically includes the following layers:
1.Rescalling Layer
1. Convolutional Layers: Extract features from the input images.
2. Pooling Layers: Reduce the dimensionality of the feature maps.
3. Flatten Layers : Flattens the output from the convolutional layers in 1D array.
4. Fully Connected Layers: Perform classification based on the extracted features.

Here is an example of a simple CNN architecture used in this project:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = models.sequential([
    layers.Rescalling(1./255,inout_shape=(image_height, image_width,1)),
    layers.Conv2D(16,1, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, padding='same' activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, padding='same' activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
# Results
- training accuracy 92.74%,loss 20.47%, validation accuracy 96.88%, validation loss 16.27%
- test accuracy 85%, loss 32%

# Acknowledgments
- Heartfelt gratitute to my team mates for successful completion of this project.

#  Result Image
- Real time prediction of the signs , input taken using web cam. 
  ![Screenshot 2024-06-15 171804](https://github.com/jojangandha-saha/ASL_SIGNS/assets/86916920/33418431-d49e-48f2-8b2b-236d0d7d68f2)
