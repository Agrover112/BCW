# Breast Cancer Wisconsin(Original)Dataset 
**Note**: *Java* implementation using *Dl4j* is added in a seperate folder

The following dataset has been taken from 
[UCI Machine Learning Repository.:]( https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))
## Performing Exploratory Analysis

![Data Distribution](https://github.com/Agrover112/BCW/blob/master/Images/Data%20Distributions.png)


 *Pearson's Correlation* of features with respect to the target variables represented here.

![Matrix](/Images/PearsonsCorrelationMatrix.png)

## Neural Network Architechture

*Adam* optimizer has been used here.

- Input Layer : ReLU

- Hidden Layer: LeakyReLU

- Output Layer : Sigmoid

![NeuralNetworkDiagram](/Images/NND.png)

## Results

| **Model**  | **Accuracy**   | **F1 Measure** |
| :------------ |:---------------:| :-----:|
| CART    | 93.80 | 95.01 |
| LR    | 94.76       |   95.87 |
| SVM | 94.76        |   95.52 |
| HVC| **95.23**        |   96.23 |
| XGBoost | **96.67**|   **97.37** |
| ANN | **95.23**|   **96.76** |
