from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

for i, y in enumerate(Y[:].T):
    print()
    print(f'Training: {data_columns[15+i]}')
    # Define the SVR model
    svr = SVR()
    # Define the grid of parameters to search
    param_grid = {
        # 'kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
        # 'C': [100, 50, 10, 1.0, 0.1, 0.01],
        # 'gamma': ['scale', 'auto']
        'kernel': ['poly', 'rbf'],
        'C': [1.0, 0.01],
    }
    # Instantiate the grid search
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, n_jobs=-1, cv=2, verbose=1)
    # Perform the grid search
    grid_result = grid_search.fit(X, y)
    # Print the best parameters found
    print("Best parameters:", grid_search.best_params_)
    # Get the best model
    best_svr = grid_search.best_estimator_
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))