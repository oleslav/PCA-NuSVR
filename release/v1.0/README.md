# DRIT
DRIT (Feature Dimension Reduction using Intermediate Model Training)

The best results that was gotten by using Tunneling_Induced_building_damage_dataset.txt.

- KFold crossvalidation 
- HistGradientBoostingRegressor as based model 
- SGTM as dimensionality analysis
- HistGradientBoostingRegressor as model to improve
- Bayesian Optimization for optimizing hyperparameters (method 'ei')

## KFold crossvalidation

5 Kfolds was used to divide dataset in to 80/20 train & test. 


## (Base model) HistGradientBoostingRegressor  

HistGradientBoostingRegressor was used as the best fitting model that was choosed after using LazyRegressor.


## (Dimensionality analysis) SGTM 

SGTM (Sequantial Geometric Transformation Model)
Author: Dr.Sc, Prof. Roman Tkachenko

For dimensionality analysis was used SGTM. It is the best model that was founded in scope of our research.

## (Improve model) HistGradientBoostingRegressor  

HistGradientBoostingRegressor was used as the best fitting model that was choosed after using LazyRegressor.
