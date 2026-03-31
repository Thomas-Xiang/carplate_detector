# Car Plate Detection using Faster R-CNN

## Overview
This project implements an object detection pipeline using Faster R-CNN with a ResNet50-FPN backbone to detect license plates in vehicle images.

The model is fine-tuned using transfer learning on a custom dataset (Car Plate Detection Dataset in COCO format).

## Features
- Pre-trained Faster R-CNN (COCO)
- Fine-tuning for license plate detection
- Data augmentation (horizontal flip)
- Evaluation using AP@0.5
- IoU-based performance analysis

## Concepts
- Object Detection (Bounding Boxes + Classification)
- Transfer Learning
- Feature Pyramid Networks (FPN)
- Backpropagation & SGD optimization

## Output
- Training & validation loss
- AP@0.5 metric
- Detection visualization

## Run with Docker
```bash
docker build -t plate-detector .
docker run -it plate-detector