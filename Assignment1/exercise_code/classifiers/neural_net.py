"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

relu = lambda x: x * (x > 0)

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.best_val_acc = 0.
        self.best_params = self.params
        self.v = {}
        self.v['W1'] = np.zeros(self.params['W1'].shape)
        self.v['W2'] = np.zeros(self.params['W2'].shape)
        self.v['b1'] = np.zeros(self.params['b1'].shape)
        self.v['b2'] = np.zeros(self.params['b2'].shape)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        layer1 = np.dot(X,W1) + b1
        activation = np.vectorize(lambda x: x * (x > 0))
        layer2 = activation(layer1)
        layer3 = np.dot(layer2, W2) + b2
        scores = layer3

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        #log_C   = np.max(layer3)
        #layer3 -= log_C
        layer3 -= np.max(layer3,axis=1,keepdims=True)
        layer3 = np.exp(layer3)
        rows = np.sum(layer3, axis=1)
        sc = layer3/rows[:,None]
        #print(layer3[np.arange(N), y] )
        logLoss = np.sum(-np.log(sc[np.arange(N), y]))/N 
 #       logLoss = np.sum(-layer3[np.arange(N), y] + np.log(rows)) / N
        loss = logLoss + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        grads['W1'] = np.zeros(W1.shape)
        grads['W2'] = np.zeros(W2.shape)
        grads['b1'] = np.zeros(b1.shape)
        grads['b2'] = np.zeros(b2.shape)
        dLogLoss = 1.0

        dLayer3 = np.transpose((layer3).T / rows)
        y_i = np.zeros(dLayer3.shape)
        y_i[np.arange(N), y] = 1
        dLayer3 -= y_i
        dLayer3 /= N
        dLayer3 *= dLogLoss

        dLayer2 = np.dot(dLayer3, W2.T)

        dLayer1 = dLayer2 * (layer1 >= 0)

        grads['W1'] = np.dot(X.T, dLayer1) + reg * W1
        grads['W2'] = np.dot(layer2.T, dLayer3) + reg * W2

        grads['b1'] = np.sum(dLayer1, axis=0)
        grads['b2'] = np.sum(dLayer3, axis=0)
    
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx,:]
            y_batch = y[idx]
            
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            #for a in self.params.keys():
#                self.v[a] = 0.90 * self.v[a] - learning_rate * grads[a]
#                self.params[a] += self.v[a]
            for a in self.params.keys():
                self.params[a] += - learning_rate * grads[a]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        scores = None
        layer1 = np.dot(X,W1) + b1
        activation = np.vectorize(lambda x: x * (x > 0))
        layer2 = activation(layer1)
        layer3 = np.dot(layer2, W2)+ b2
        scores = layer3
        y_pred = np.argmax(scores, axis=1)
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 
    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################


    input_size = X_train.shape[1]
    hidden_size_set = [32,64,128,256]
    #learning_rate   = [ 1e-3, 1e-4, 1e-5]
    learning_rate   = [ 1e-3 ]
    #learning_rate_decay = [0.99, 0.97, 0.95, 0.90 ]
    regularization_strength = [ 0.25, 0.5, 1.0, 1.5]
    num_classes = 10

    loss_history_dict = {}
    best_acc = -np.inf
    for hs in hidden_size_set:
        for lr in learning_rate:
            print("learning rate",lr)
            print("Hidden Unit Size", hs)
            net = TwoLayerNet(input_size, hs, num_classes)

        # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,num_iters=500, batch_size=500,learning_rate=lr, learning_rate_decay=0.95,reg=1.0, verbose=False)
        
            plt.plot(stats['loss_history'])
            plt.xlabel('Iteration number')
            plt.ylabel('Loss value')
            plt.show()

            # Predict on the validation set
            val_acc = (net.predict(X_val) == y_val).mean()
            #print("hs=%d,lr=%f"%(hs,lr))
            print('Validation accuracy: ', val_acc)
            print(net.best_val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_net = deepcopy(net)


    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
