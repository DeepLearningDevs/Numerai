# Numerai
Model for https://numer.ai/ tournaments. 

Model consists of an assembly of several neural networks, logistic regression, xgboost, lightGBM and ExtraTreeClassifier.

Assembly is done by stacking, there is also a script for higher level feature extraction and data exploration notebook.

Next step would be grid search for optimal hyper-parameters and try to use LSTM on eras as time-series classification.
