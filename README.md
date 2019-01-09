# HyperparameterSearchingForModels
This project is about writing a set of codes that search hyper parameter sets for a given model that yields the best model performance.

Feature engineering is done after reading famous 'introduction to stacking method' kernel from Kaggle. I made some changes as I felt it's necessary.

I used 6 differnet models; famous xgb, neural network, factorimation machines, logistic regression, k-nearest neighbor, and random forests. This code consists of 3 steps.

1. Creating range of parameter spaces for each models and searching for the set of model hyper parameters that yields the best model performance (k-fold shuffle spliting). 
2. Pooling models to create second level features and removing highly correlated features.
3. Passing the test data several stage of model stacking.

Parameter ranges were determined after quick research how people use them and its definitions. 
Result was submitted to Kaggle Titanic competition page and score for their test data was 0.7979.
