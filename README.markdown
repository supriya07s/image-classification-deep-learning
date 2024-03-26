
# Fire Image Classification using Deep Learning

- The Fire Image Classification project aims to develop a robust machine learning model capable of detecting the presence of fire in images. 
- Leveraging Amazon SageMaker, this project offers a scalable and efficient solution for real-time inference on image data.

## Installation and Setup

- We've used the AWS Sagemaker Tensorflow 2.12 Image with Python 3.10 GPU.
- All the required packeges are mentioned in the requirements.txt file.
- To setup the environment select the above mentioned image and install all packages by running the commands:
    `! pip install requirements.txt`
- The dataset comtains of three folders with 10800 images each of fire, non fire and smoke categories in train data and 3500 images in test data.
    - We're using only 30% of the data due to computational constraints.
    - We're only using images from fire and non-fire categories since adding smoke dataset in model is not giving good results.

## Motivation

- The detection of fire in images is crucial for various applications, including fire monitoring systems, surveillance cameras, and firefighting efforts.
- Traditional methods for fire detection often rely on manual inspection or rule-based algorithms, which may not be scalable or accurate. 
- By harnessing the power of machine learning and computer vision techniques, this project seeks to automate the process of fire detection, enabling faster response times and improved safety measures.

## Approach

- The project adopts a deep learning approach, specifically utilizing Convolutional Neural Networks (CNNs), to classify images as containing fire or not. 
- CNNs are well-suited for image classification tasks, as they can automatically learn relevant features from raw pixel data. 
- The model architecture consists of multiple convolutional layers followed by pooling layers to extract hierarchical features and learn spatial representations. - The trained model is saved in S3 and can be called to do inferencing on the dataset.

## Model Choice
### Reasoning

- Convolutional Neural Networks (CNNs) are well-suited for image classification tasks due to their ability to automatically learn hierarchical features from images.
- CNNs have been proven effective in various image classification tasks, including detecting objects and scenes in images.
- The chosen model architecture includes convolutional layers followed by pooling layers to extract relevant features from input images and learn spatial hierarchies.

## Model Architecture Description:

- Input Layer: Accepts input images with dimensions 256x256 pixels and 3 channels (RGB).
- Convolutional Layers: Three sets of convolutional layers with increasing filter sizes (32, 64, and 128). Each convolutional layer is followed by a ReLU activation function to introduce non-linearity.
- Pooling Layers: Max pooling layers are applied after each convolutional layer to reduce the spatial dimensions of the feature maps while retaining the most important information.
- Flatten Layer: Converts the 3D feature maps into a 1D vector to be fed into the fully connected layers.
- Fully Connected Layers: Two fully connected (dense) layers with 512 and 1 neuron(s) respectively. ReLU activation function is applied to the first fully connected layer to introduce non-linearity, and a sigmoid activation function is applied to the final layer to produce a binary classification output (fire or non-fire).

## Discussion of Future Work:

- Deploy the trained model as a web service or mobile application for real-time fire detection in various scenarios.
- Integrate the model with existing fire monitoring systems, surveillance cameras, and IoT devices to enhance fire detection capabilities.

## Performance Evaluation:

### Training Accuracy and Loss:
- During the training process, we monitor the training accuracy and loss to assess how well the model is learning from the training data. We aim for high accuracy and low loss values, indicating that the model is effectively capturing patterns and making accurate predictions.

### Validation Accuracy and Loss:
- We also evaluate the model's performance on a separate validation dataset to ensure that it generalizes well to unseen data. Monitoring validation accuracy and loss helps us identify potential overfitting or underfitting issues and adjust the model accordingly.

### Test Accuracy and Other Metrics:
- Finally, we evaluate the trained model on a held-out test dataset to obtain an unbiased estimate of its performance. We calculate metrics such as accuracy, precision, recall, and F1 score to measure the model's effectiveness in detecting fire in images.

![Final Model Performance with Accuracy](newplot.png)

