#9793565

import numpy as np

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new column with 1s.

    :param X: design matrix
    :type X: np.array
    :param y: regression targets
    :type y: np.array
    :return: prediction
    :rtype: np.array
    """
    # complete a função e remova a linha abaixo
    X = np.insert(X, 0, 1.0, axis=1)
    X_transpose = np.transpose(X)
    mult_Xtranspose_X = np.matmul(X_transpose, X)
    pseudo_inverse = np.linalg.inv(mult_Xtranspose_X)
    mult_pseudoinverse_Xtranspose = np.matmul(pseudo_inverse, X_transpose)
    w =  np.matmul(mult_pseudoinverse_Xtranspose, y)
    w_transpose = np.transpose(w)

    prediction = np.empty(shape=(np.shape(X)[0], np.shape(y)[1]))
    tmp_index = 0
    for sample in X:
        prediction[tmp_index] = np.dot(w_transpose, sample)
        tmp_index+=1

    return prediction
