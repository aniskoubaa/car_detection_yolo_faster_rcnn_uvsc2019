# Car Detection using Unmanned Aerial Vehicles: Comparison between Faster R-CNN and YOLOv3

* This repo contains the dataset and link to the source used in the paper  "Car Detection using Unmanned Aerial Vehicles: Comparison between Faster R-CNN and YOLOv3"

#### Links

* [paper](https://arxiv.org/abs/1812.10968) - Link to the paper
* [video](https://www.youtube.com/watch?v=rlPUhJmKcv4) - Video of the paper

#### Abstract
* Unmanned Aerial Vehicles are increasingly being used in surveillance and traffic monitoring thanks to their high mobility and ability to cover areas at different altitudes and locations. One of the major challenges is to use aerial images to accurately detect cars and count them in real-time for traffic monitoring purposes. Several deep learning techniques were recently proposed based on convolution neural network (CNN) for real-time classification and recognition in computer vision. However, their performance depends on the scenarios where they are used. In this paper, we investigate the performance of two state-of-the-art CNN algorithms, namely Faster R-CNN and YOLOv3, in the context of car detection from aerial images. We trained and tested these two models on a large car dataset taken from UAVs. We demonstrated in this paper that YOLOv3 outperforms Faster R-CNN in sensitivity and processing time, although they are comparable in the precision metric.

#### Citation
```
@misc{benjdira2018car,
    title={Car Detection using Unmanned Aerial Vehicles: Comparison between Faster R-CNN and YOLOv3},
    author={Bilel Benjdira and Taha Khursheed and Anis Koubaa and Adel Ammar and Kais Ouni},
    year={2018},
    eprint={1812.10968},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

#### Dataset used in the paper
* To perform the experimental part of our study, we built a UAV imagery dataset divided into a training set and a test set. 
* We tried to collect cars from different environments andscales to assure the validity of our experiment and to test the genericity of the algorithms. For example, some images are taken from an altitude of 55m and others are taken from above 80m.
* The training set contains 218 images and 3,365 instances of labeled cars. The test set contains 52 images and 737 instances of cars. 
* This dataset was collected from:
  - images taken by an UAV flown above Prince Sultan University campus. This is what we provide in this repository. We provided three folders:
    - The images: The image ataken by the UAV flown above Prince Sultan University campus.
    - The labels for these images in XML format.
    - scripts to convert the labels from the xml format to the VOC format and to the YOLO format.
  - from an open source dataset available in Github here: https://github.com/jekhor/aerial-cars-dataset. We used the images and their labels.
  
### Training Faster R-CNN
* To train the model Faster R-CNN on the constructed dataset, we used [Tensoflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
* A quick start file is provided to run how the run Tensorflow Object Detection API on a chosen dataset: [Running Tensorflow Object Detection on Pets Dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md)
* We used the pretrained weights for Faster R-CNN model based on the Feature Extractor Inception v2 and pretrained on COCO dataset. The model is provided here: [Tensorflow Object Detection API model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

### Training YOLO v3
* To train the model YOLO v3, we used the default YOLO v3 implementation provided here: [YOLO v3](https://pjreddie.com/darknet/yolo/).
* After building the training binary for YOLO: run the following command:
```
./darknet detector train cfg/coco.data cfg/yolov3.cfg backup/yolov3.backup -gpus 0,1,2,3
```
### Car detection using Faster R-CNN
![alt text](https://github.com/aniskoubaa/car_detection_yolo_faster_rcnn_uvsc2019/blob/master/car-detection-faster-r-cnn.jpg)

### Car detection using YOLO v3
![alt text](https://github.com/aniskoubaa/car_detection_yolo_faster_rcnn_uvsc2019/blob/master/car-detection-yolo-v3.png)








