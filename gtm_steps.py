import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import math
import time

class MultiLabelGTM(BaseEstimator, ClassifierMixin):

    def __init__(self, name="MultiLabelGTM", number_of_steps = 1000, n_iter=5, center_of_mass=False, verbose=0):
        self.name = name
        self.number_of_steps = number_of_steps
        self.n_iter = n_iter
        self.center_of_mass = center_of_mass
        self.verbose = verbose

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y, multi_output=True)

        start_time = time.time()

        m = X.shape[1]
        self.outputs_count = Y.shape[1]
        step = 0

        if self.number_of_steps > m and self.verbose > 1:
          print('Number of steps are greater than input features!')

        if self.verbose > 0:
            print(f'Total input features: {m}')

        centered_X = X
        centered_Y = Y

        com_x = np.zeros(m)
        com_y = 0 # np.zeros(self.outputs_count)

        if self.center_of_mass:
            com_x = np.mean(X, axis=0)
            com_y = np.mean(Y)
            centered_X = X - com_x
            centered_Y = Y - com_y

        self.com_x = com_x
        self.com_y = com_y

        self.basic_instances = []
        self.b_instances_x = []
        self.b_instances_y = []

        while step < m:
            iter = 0

            # TODO change
            i = np.argmax(np.sum(centered_X ** 2, axis=1))

            initial_instance = centered_X[i,]

            basic_instance = initial_instance

            while iter < self.n_iter:
                vector_relations = np.sum(np.multiply(centered_X, basic_instance), axis=1) \
                                   / np.sum(basic_instance ** 2)
                basic_instance = np.sum(np.multiply(centered_X, vector_relations[:, np.newaxis]), axis=0) \
                                 / np.sum(vector_relations ** 2)
                iter = iter + 1

            k1_values = np.sum(np.multiply(centered_X, basic_instance), axis=1) \
                        / np.sum(basic_instance ** 2, axis=0)

            k2_values = np.sum(np.multiply(centered_X, basic_instance), axis=1) \
                        / math.sqrt(np.sum(basic_instance ** 2, axis=0))

            if step == self.number_of_steps:
                self.num_steps = step
                if self.verbose > -1:
                    print("Break, step = " + str(step))
                break

            b_instance_x = np.sum(np.multiply(centered_X, k1_values[:, np.newaxis]), axis=0) \
                           / np.sum(k1_values ** 2, axis=0)

            b_instance_y = np.sum(np.multiply(centered_Y, k1_values[:, np.newaxis]), axis=0) \
                           / np.sum(k1_values ** 2, axis=0)

            centered_X = np.subtract(centered_X, np.multiply(k1_values[:, np.newaxis], b_instance_x))
            centered_Y = np.subtract(centered_Y, np.multiply(k1_values[:, np.newaxis], b_instance_y))

            self.basic_instances.append(basic_instance)
            self.b_instances_x.append(b_instance_x)
            self.b_instances_y.append(b_instance_y)

            step = step + 1
            if self.verbose > 1:
                print("Step: " + str(step))

        self.num_steps = step
        if self.verbose > 0:
            print("--- %s seconds ---" % (time.time() - start_time))

        return self

    def predict(self, X):

        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        centered_X = X

        if self.center_of_mass:
            com_x = self.com_x
            centered_X = X - com_x

        k1_values_a = []

        for step in range(0, self.num_steps):
            k1_values = np.sum(np.multiply(centered_X, self.basic_instances[step]), axis=1) \
                        / np.sum(self.basic_instances[step] ** 2, axis=0)
            centered_X = np.subtract(centered_X, np.multiply(k1_values[:, np.newaxis], self.b_instances_x[step]))
            k1_values_a.append(k1_values)

        Y = np.zeros((X.shape[0], self.outputs_count))
        for step in reversed(range(0, self.num_steps)):
            k1_values_vector = np.array(k1_values_a[step])
            Y = np.multiply(k1_values_vector[:, np.newaxis], self.b_instances_y[step]) + Y

        Y = Y + self.com_y

        return Y

    def map_to_principal_component(self, X):
        X = check_array(X)

        centered_X = X

        if self.center_of_mass:
            com_x = self.com_x
            centered_X = X - com_x

        k1_values_a = []

        for step in range(0, self.num_steps):
            k1_values = np.sum(np.multiply(centered_X, self.basic_instances[step]), axis=1) \
                        / np.sum(self.basic_instances[step] ** 2, axis=0)
            centered_X = np.subtract(centered_X, np.multiply(k1_values[:, np.newaxis], self.b_instances_x[step]))
            k1_values_a.append(k1_values)

        return np.asarray(k1_values_a, dtype=np.float32).transpose()


def to_multylabel(x, labels):
    result = np.full((len(x), len(labels)), -1)
    result[np.arange(len(x)), x] = 1
    return result

