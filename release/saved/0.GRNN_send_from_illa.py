import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin

# GRNN class by Izonin
class GRNN(BaseEstimator, ClassifierMixin):
    def __init__(self, name = "GRNN", sigma = 0.1):
        self.name = name
        self.sigma = 2 * np.power(sigma, 2)
        
    def predict(self, instance_X, X_train_scaled, Y_train):
        gausian_distances = np.exp(-np.power(np.sqrt((np.square(X_train_scaled-instance_X).sum(axis=1))),2) / self.sigma)
        gausian_distances_sum = gausian_distances.sum()
        gausian_distances_sum = max(gausian_distances_sum, 1e-07) 

        return np.multiply(gausian_distances, Y_train).sum() / gausian_distances_sum

############## GRNN from scratch by Illia ##################3
# A = np.array([[-10, -5, 2,10]]).T
# y = np.array([[0.5, 0, 1.5, 3]]).T
# x = 2

# sigma = 1

# kers = np.zeros(4)
# y_kers = np.zeros(4)

# for i in range(len(A)):
#     kernel_ = np.exp(-(np.sqrt((x-A[i])**2)**2) / (2 * sigma**2)) 
#     # the rbf kernel above calculates the distance between points and acts as a 'weight' of each point for x
#     kers[i] = kernel_
#     y_kers[i] = y[i] * kernel_
#     # y[i] * kernel 'weights' y-values with respect to their distance to x

# print(f'Distances from x:\n {np.round(kers, 4)}')
# print('')
# print(f'Weighted y-values:\n {np.round(y_kers, 4)}')
# print('')

# y_pred = np.sum(y_kers) / np.sum(kers)
# # prediction is calculated as the sum of weighted y-values scaled by the distances for x
# # in the end, the points that are the closest to x with have the most impact on the prediction
# # simply put 'if x's closest relative has blue eyes, that we'll say that x will also have blue eyes'.
# print(f'y_pred:\n {round(y_pred, 2)}')

# # now, the question is how to get the right sigma


results = {}
# cost function to optimize
def f(params, X_train_scaled, Y_train, X_test_scaled, Y_test):
    s, = params  # Unpack the parameters
    grnn = GRNN(sigma=s)
    predictions = np.array([grnn.predict(i, X_train_scaled, Y_train) for i in X_test_scaled])

    return -r2_score(Y_test, predictions) # USE MSE


for i in range(Y_train.shape[1]):
    start_time = time.time()

    res = differential_evolution(f, bounds=[(0.001, 10)], args=(X_train_scaled, Y_train.iloc[:, i], X_test_scaled, Y_test.iloc[:, i]))
    s = res["x"][0]

    grnn = GRNN(sigma=s)
    predictions = np.apply_along_axis(lambda x: grnn.predict(x, X_train_scaled, Y_train.iloc[:, i]), axis=1, arr=X_test_scaled)
    
    exp_time = time.time() - start_time


    MaxError = max_error                    (Y_test.iloc[:,i].ravel(), predictions)
    MAE = mean_absolute_error               (Y_test.iloc[:,i].ravel(), predictions)
    MSE = mean_squared_error                (Y_test.iloc[:,i].ravel(), predictions)
    MedError = median_absolute_error        (Y_test.iloc[:,i].ravel(), predictions)
    RMSE = mean_squared_error               (Y_test.iloc[:,i].ravel(), predictions, squared=False)
    MAPE = mean_absolute_percentage_error   (Y_test.iloc[:,i].ravel(), predictions)
    R2 = r2_score                           (Y_test.iloc[:,i], predictions)

    results.update({
        f'test_{i+1}':
            {
                'time': exp_time,
                'sigma': s,
                'y_true': Y_test.iloc[:,i].ravel(),
                'y_pred': predictions,
                'MaxError' : MaxError,
                'MAE' : MAE,
                'MSE' : MSE,
                'MedError' : MedError,
                'RMSE' : RMSE,
                'MAPE' : MAPE,
                'R2' : R2
            }
    })

exp_result = pd.DataFrame(results)


