# Modelling-with-CICIDS2017

The purpose of this project is to compare the performance between a vanilla ANN and an ANN utilising feature maps from the bottleneck of an Autoencoder. 
The models used are based on [winning solution](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348) of Kaggle Competition Jane Street financial modelling.

## Dataset
As the title suggests CICIDS 2017 dataset was used and in a nutshell, the dataset is a collection of cyber security attacks which were simulated over 5 days from 9 AM to 5 PM. 
The dataset consists of 14 types of network attacks and 1 benign class. The dataset has around 70 features and each feature showcases the detail of particular aspects,
such as the port number on which the attack was conducted.   
The dataset was available on kaggle and this eased up the process of experimentation. <br>
Link for the dataset : [Dataset Link](https://www.kaggle.com/cicdataset/cicids2017)

## Code Structure
`preprocessing.py` : Contains the necessary code to clean and preprocess the csv files in the dataset. The csv files are not present in this repository due to its size.
This will also output `train_df.csv` and `valid_df.csv` which will be used for training the models. Set the variable `WORK_DIR` as the directory where you download the csv files from kaggle.

`getLists.py` : Contains 2 python list with the train columns and the class labels. These are commonly needed across all files.

`evaluate_model.py` : Contains the script for evaluating the model with ROC_AUC, accuracy score, sensitivity and recall. 

`ANN.py` : Contains the model for  ANN and train script for the same.

`AutoEncoder+ANN.py` : Contains the model for  Autoencoder + ANN and train script for the same.

The last 2 files are the ones used to train the models.

## Results
| Model Type | Accuracy Score | ROC AUC |
|------------|----------------|---------|
| ANN        | 0.99           | 0.85    |
| ANN +AE    | 0.98           | 0.88    |


Based on the results, the ANN + Autoencoder model performed the best. While analyzing the results from the valid set it was observed that the auto encoder plus
ANN model performed better on classes with lower occurrences whereas the predictions made by the ANN model predicted most of these threats to be benign. The 
same was done using a confusion matrix. Although the high score on the metrics typically tend to indicate a good model, 50% of the dataset belongs to Benign 
class which impacts the learning of the ANN model. The reason the autoencoder performs well could be attributed to the feature extraction from the bottleneck
which combined with the original features provide a perfect base for the ANN to learn. 
