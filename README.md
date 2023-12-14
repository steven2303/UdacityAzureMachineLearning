# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This project contains data about bank marketing campaigns, aiming to predict customer subscription to term deposits. 

The metric employed for assessing the performance of the various models was their accuracy. The most effective model emerged from the use of AutoML, with the VotingEnsemble method standing out as the top performer with an accuracy of 0.9183.

## Scikit-learn Pipeline

The pipeline involves data preprocessing, hyperparameter tuning, and logistic regression classification. 
- Data Preprocessing: This includes encoding categorical variables. Our goal in this phase is to transform the raw data into a format that can be effectively used by the logistic regression model.
- Hyperparameter Tuning: After preprocessing, we focus on hyperparameter tuning, which is crucial for optimizing the model's performance. We employ Azure ML's HyperDrive to find the best values for the hyperparameters (C - the inverse of regularization strength, and max_iter - maximum number of iterations to converge) from a defined range.
- Logistic Regression Classification: The final step is the application of the logistic regression algorithm. Logistic regression is a widely-used method for binary classification tasks. The tuned hyperparameters from the previous step are used here to train the logistic regression model.

RandomParameterSampling benefits:
- It's a faster, less demanding way to try out different settings for the model.
- It randomly tries out a bunch of settings, which helps find the good ones without needing to test everything.

BanditPolicy benefits:
- Stops training that isn't doing well early, so you don't waste time or computer power.
- Focuses on the training that's working well, making the whole process more effective.

## AutoML
AutoML produced a diverse range of models, For each model, it shows:
- The type of model (like LightGBM, XGBoostClassifier, etc.).
- How long it took to test each model.
- The accuracy metric of each model, with the best one so far highlighted.
The best model found was a VotingEnsemble with an accuracy of 0.9183. This model combines predictions from multiple other models to get better results.

## Pipeline comparison
The HyperDrive logistic regression model had slightly lower accuracy than the AutoML VotingEnsemble. The manual tuning approach in HyperDrive contrasts with AutoML's automated, exhaustive search across various models and hyperparameters. This comprehensive approach likely led to AutoML's superior performance.

## Future work
Improvements for future experiments include exploring more complex hyperparameter grids in HyperDrive and leveraging AutoML's advanced features, such as feature engineering and deep learning algorithms. These improvements could potentially uncover more optimal model configurations and enhance predictive performance.


