"""
Includes methods for training and predicting using naive Bayes.
Designed for use with binary predictors, but can be easily generalized.
@author Neil Gutkin
"""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype model: dict
    """

    labels = np.unique(train_labels)

    d, n = train_data.shape
    num_classes = labels.size

    # Set LaPlace smoothing parameter. 
    # As alpha increases, the probabilities tend towards the uniform distribution.
    # If there is little need for smoothing, setting alpha low will typically boost accuracy.
    alpha = 1

    # Initialize prior and likelihood arrays
    p_y = np.zeros(num_classes, dtype=float)
    likelihoods = np.zeros((d, num_classes), dtype=float)

    # Loop over each class
    for c in range(num_classes):
        # Get documents with current class label
        class_data = train_data[:,np.where(train_labels == c)[0]]
        # Get number of examples of current class
        m = class_data.shape[1]

        # Get current prior (w/ LaPlace Smoothing)
        p_y[c] = (m + alpha) / (n + alpha*num_classes)
        # Get class conditional probabilities (w/ LaPlace Smoothing)
        likelihoods[:,c] = (np.sum(class_data, axis=1) + alpha) / (m + 2*alpha)

    # Combine priors and likelihoods into model dictionary
    model = dict()
    model['likelihood'] = likelihoods
    model['prior'] = p_y
    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    d, n = data.shape
    # Convert priors and likelihoods to log-scale
    prior = np.log(model['prior'])
    likelihood = np.log(model['likelihood'])
    num_classes = likelihood.shape[1]
    prediction = np.zeros(n, dtype=float)

    # Each cell i,j (i < num_classes, j < n) corresponds to log(prior_i) + log(conditionals_ij)
    # In other words, each cell is the numerator of log(P(Y=i|X_j))
    probs = np.zeros((num_classes, n))
    # Find the sums of logged feature|class conditionals for each document
    probs = np.matmul(likelihood.T, data)
    # Add the appropriate logged prior to each row
    probs = np.add(prior.reshape(num_classes,1), probs)

    # Find index of maximum probability term for each document
    prediction = np.argmax(probs, axis=0)
    return prediction
