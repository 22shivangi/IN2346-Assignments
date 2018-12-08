import pickle as pickle
import os


def save_model(modelname, data):
    dir = 'models'
    model = {modelname: data}
    if not os.path.exists(dir):
        os.makedirs(dir)
    pickle.dump(model, open(dir + '/' + modelname + '.p', 'wb'))


def save_fully_connected_net(classifier):
    modelname = 'fully_connected_net'
    save_model(modelname, classifier)
