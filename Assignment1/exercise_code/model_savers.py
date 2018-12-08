"""Helper Functions for Saving Model Files."""
import os
import pickle as pickle


def save_model(modelname, data):
    """Save given model with the given name."""
    directory = 'models'
    model = {modelname: data}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model, open(directory + '/' + modelname + '.p', 'wb'))


def save_softmax_classifier(classifier):
    """Wraps model saving for the softmax_classifier model."""
    modelname = 'softmax_classifier'
    save_model(modelname, classifier)


def save_two_layer_net(classifier):
    """Wraps model saving for the two_layer_net model."""
    modelname = 'two_layer_net'
    save_model(modelname, classifier)


def save_feature_neural_net(classifier):
    """Wraps model saving for the feature_neural_net model."""
    modelname = 'feature_neural_net'
    save_model(modelname, classifier)
