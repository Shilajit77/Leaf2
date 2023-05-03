# Leaf2
# Developing a system for Leaf Disease Detection<br>

Link for Demonstration: https://shilajit77-leaf2-trying-cy1r0u.streamlit.app/

Name of Students:
1. Shilajit Banerjee(M22CS062)
2. Dattatreyo Roy(M22CS060)

## Overview

The main objective of this project is proposing a system for detecting leaf diseases using different deep learning models. The system utilizes these deep learning models to classify the images of leaves into healthy or diseased categories and identify its disease. However, since a single deep learning model may not be always accurate and a differently and better trained model may give better results, so we have used the concept of ensemble learning to improve accuracy, precision and recall. We have created an interface which allows users to use different model to detect the diseases. The results demonstrate that the ensembled model achieves comparable accuracy,precision and recall to the larger teacher model, while requiring less computational resources and time. 


## Objectives

1. Developing a leaf disease detection system using deep learning models: The main objective of the project is to develop a system that can accurately detect leaf diseases using deep learning models. This would involve training the models on a large dataset of images of healthy and diseased leaves.
2. Improving the accuracy, precision, and recall of the detection system through the Ensemble learning which can help improve the overall performance of the system by combining the predictions of multiple models.
3. Comparing the performance of the ensembled model to the model where we aim to compare the performance of the ensembled model to determine whether the ensemble approach is more effective in improving the performance of the detection system.

<p align="center">
  <img src="Images/ensemble1.png" alt="Ensemble model" width="500"/><br>
 </p>
 
 
5. Developing a user-friendly interface for the system: The project could also aim to develop a user-friendly interface that allows users to choose different models for detecting diseases, providing flexibility and ease of use. This would involve designing and implementing an interface that is intuitive and easy to navigate.
6. Another objective is to optimize the computational resources and time required for the system to run. This would involve finding ways to reduce the computational resources and time required while maintaining the accuracy and performance of the system.


## Dataset

We have taken the dayaset from Kaggle. The plant leaves dataset on Kaggle consists of 4,503 images of 12 plant species, namely Mango, Arjun, Alstonia Scholaris, Guava, Bael, Jamun, Jatropha, Pongamia Pinnata, Basil, Pomegranate, Lemon, and Chinar. The images have been labeled into two classes - healthy and diseased. The plants were named from P0 to P11 and the dataset was divided into 22 subject categories ranging from 0000 to 0022. The images labeled with categories 0000 to 0011 represent the healthy class, while those labeled with categories 0012 to 0022 represent the diseased class.
The dataset contains 2,278 images of healthy leaves and 2,225 images of diseased leaves.<br>
Follow this link to get an overview of the dataset:


## Approach to the problem:

In this project, we have developed the leaf disease detection system using three well-known deep learning models such as DenseNet121, ResNet101, and VGG16, to classify images of leaves into healthy or diseased categories and identify the specific disease. Having experimented with the individual models, we have found out the results. Now, to improving the accuracy, precision, and recall of the detection system, we have tried to use the Ensemble learning which can help improve the overall performance of the system by combining the predictions of multiple models.


## Interface

we have developed the user-friendly interface for this system which allows the users to choose different models for detecting diseases, providing flexibility and ease of use. This would involve designing and implementing an interface that is intuitive and easy to navigate. The users takes a look at the different diseases. Now, if they select any model and enter the leaf image with disease, it will show the detected label for that leaf.

<p align="center">
  <img src="Images/ee1.jpg" alt="Ensemble model" width="400"/><br>
 </p>
 
 
 <table>
  <tr>
    <td><img src="Images/m1.jpg" alt="Ensemble model" width="500"/></td>
    <td><img src="Images/m2.jpg" alt="Ensemble model" width="500"/></td>
    <td><img src="Images/m3.jpg" alt="Ensemble model" width="500"/></td>
  </tr>
</table>

<p align="center">
  <img src="Images/emb.jpg" alt="Ensemble model" width="400"/><br>
 </p>

## Results

Without Smoting: 

| Model         | Accuracy | Precision | Recall | F1 Score |
| -------------| --------| --------- | ------ | -------- |
| DenseNet 121  | 0.75    | 0.77      | 0.75   | 0.69     |
| Resnet 101    | 0.77    | 0.81      | 0.77   | 0.77     |
| VGG16         | 0.67    | 0.65      | 0.67   | 0.65     |


With Smoting: 


| Model         | Accuracy | Precision | Recall | F1 Score |
| -------------| --------| --------- | ------ | -------- |
| DenseNet 121  | 0.77    | 0.79      | 0.77   | 0.78     |
| Resnet 101    | 0.74    | 0.78      | 0.73   | 0.73     |
| VGG16         | 0.69    | 0.64      | 0.69   | 0.70     |


## Conclusion
