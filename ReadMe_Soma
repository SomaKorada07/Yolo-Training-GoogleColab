# Architecture of YOLO:

YOLO divides the input image into an **S**×**S** grid. Each grid cell predicts only **one** object. For example, the yellow grid cell below tries to predict the “person” object whose center (the blue dot) falls inside the grid cell.

![1](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/1.jpeg)

Each grid cell predicts a fixed number of boundary boxes. In this example, the yellow grid cell makes two boundary box predictions (blue boxes) to locate where the person is.

![2](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/2.jpeg)

For each grid cell,

- it predicts **B** boundary boxes and each box has one **box confidence score**,
- it detects **one** object only regardless of the number of boxes B,
- it predicts **C** **conditional class probabilities** (one per class for the likeliness of the object class).

Each boundary box contains 5 elements: (*x, y, w, h*) and a **box confidence score**. The confidence score reflects how likely the box contains an object (**objectness**). The **conditional class probability** is the probability that the detected object belongs to a particular class (one probability per category for each cell). So, YOLO’s prediction has a shape of (S, S, B×5 + C) = (7, 7, 2×5 + 20) = (7, 7, 30).

![3](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/3.jpeg)

The major concept of YOLO is to build a CNN network to predict a (7, 7, 30) tensor. It uses a CNN network to reduce the spatial dimension to 7×7 with 1024 output channels at each location. YOLO performs a linear regression using two fully connected layers to make 7×7×2 boundary box predictions (the middle picture below). To make a final prediction, we keep those with high box confidence scores (greater than 0.25) as our final predictions (the right picture).

![4](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/4.png)

![5](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/5.png)

YOLO has 24 convolutional layers followed by 2 fully connected layers (FC). Some convolution layers use 1 × 1 reduction layers alternatively to reduce the depth of the features maps. For the last convolution layer, it outputs a tensor with shape (7, 7, 1024). The tensor is then flattened. Using 2 fully connected layers as a form of linear regression, it outputs 7×7×30 parameters and then reshapes to (7, 7, 30), i.e. 2 boundary box predictions per location.

A faster but less accurate version of YOLO, called Fast YOLO, uses only 9 convolutional layers with shallower feature maps.

YOLO can make duplicate detections for the same object. To fix this, YOLO applies non-maximal suppression to remove duplications with lower confidence. Following are the steps taken for implementing NMS:

1. Sort the predictions by the confidence scores.

2. Start from the top scores, ignore any current prediction if we find any previous predictions that have the same class and IoU > 0.5 with the current prediction.

3. Repeat step 2 until all predictions are checked.

   

# Loss Function:

YOLO predicts multiple bounding boxes per grid cell. To compute the loss for the true positive, we only want one of them to be **responsible** for the object. For this purpose, we select the one with the highest IoU (intersection over union) with the ground truth. This strategy leads to specialization among the bounding box predictions. Each prediction gets better at predicting certain sizes and aspect ratios.

YOLO uses sum-squared error between the predictions and the ground truth to calculate loss. The loss function composes of:

- the **classification loss**.
- the **localization loss** (errors between the predicted boundary box and the ground truth).
- the **confidence loss** (the objectness of the box).

**Classification loss**

If *an object is detected*, the classification loss at each cell is the squared error of the class conditional probabilities for each class:

![6](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/6.png)



**Localization loss**

The localization loss measures the errors in the predicted boundary box locations and sizes. We only count the box responsible for detecting the object.

![7](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/7.png)

We do not want to weight absolute errors in large boxes and small boxes equally. i.e. a 2-pixel error in a large box is the same for a small box. To partially address this, YOLO predicts the square root of the bounding box width and height instead of the width and height. In addition, to put more emphasis on the boundary box accuracy, we multiply the loss by λ*coord* (default: 5).

**Confidence loss**

If *an object is detected in the box*, the confidence loss (measuring the objectness of the box) is:

![8](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/8.png)

If *an object is not detected in the box*, the confidence loss is:

![9](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/9.png)

Most boxes do not contain any objects. This causes a class imbalance problem, i.e. we train the model to detect background more frequently than detecting objects. To accommodate this, we weight this loss down by a factor λ*noobj* (default: 0.5).

**Loss**

The final loss adds localization, confidence and classification losses together.

![10](https://github.com/SomaKorada07/TrainYourOwnYOLO/blob/master/10.png)

# Versions of YOLO:

- YOLO - Input image is divided into 7X7 grid cells and number of templates is 2. Limitation is objects closely placed are not detected as one grid cell detects only one object.
- YOLOV2 - Input image is divided into 13X13 grid cells and number of templates is 5. Added Batch Normalization in the convolution layers.
- YOLO9000 - Input image is divided into 26X26 grid cells and number of templates is 5. Mainly used for satellite images which are very small. YOLO9000 extends YOLO to detect objects over 9000 classes using hierarchical classification with a 9418 node WordTree.
- YOLOV3 - YOLOv3 replaces the softmax function with independent logistic classifiers to calculate the likeliness of the input belongs to a specific label. Instead of using mean square error in calculating the classification loss, YOLOv3 uses binary cross-entropy loss for each label. This also reduces the computation complexity by avoiding the softmax function. YOLOv3 applies k-means cluster.

# Reference - https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
