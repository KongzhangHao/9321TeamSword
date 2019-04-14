from sklearn import svm


def train_svm(parameters, x_train, y_train):
    ## Populate the parameters...
    gamma = parameters['gamma']
    C = parameters['C']
    kernel = parameters['kernel']
    degree = parameters['degree']
    coef0 = parameters['coef0']

    ## Train the classifier...
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
    # assert x_train.shape[0] <=541 and x_train.shape[1] <= 5720
    clf.fit(x_train, y_train)
    return clf

