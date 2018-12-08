"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)  # this is the prediction of training sample i, for each class
        scores -= np.max(scores) #Regularize the scores by subtracting max values

        # calculate the probabilities that the sample belongs to each class
        probabilities = np.exp(scores) / np.sum(np.exp(scores))

        # loss is the log of the probability of the correct class
        req_prob = probabilities[y[i]]
        loss += -np.log(req_prob)

        probabilities[y[i]] -= 1 # calculate p-1 and later we'll put the negative back
    
        # dW is adjusted by each row being the X[i] pixel values by the probability vector
        for j in range(num_classes):
          dW[:,j] += X[i,:] * probabilities[j]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    num_train = X.shape[0]
    scores = X.dot(W) # NxD * DxC = NxC
    scores -= np.max(scores,axis=1,keepdims=True)
    probabilities = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
    correct_class_probabilities = probabilities[range(num_train),y]

    loss = np.sum(-np.log(correct_class_probabilities)) / num_train
    # that was supposed to summarize across classes that aren't classified correctly
    # so now we need to subtract 1 class for each case (a total of N) that are correctly classified
    loss += 0.5 * reg * np.sum(W*W) 

    probabilities[range(num_train),y] -= 1
    dW = X.T.dot(probabilities) / num_train
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)

def compute_accuracy(y, y_pred):
    return np.mean(y == y_pred)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7,1e-6,1e-5, 1e-4, 1e-3]
    regularization_strengths = [1e-1,1e1,5e1,1e2,1e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    iters = 1500
    for lr in learning_rates:
        for reg in regularization_strengths:
            # train softmax classifier
            print("**********************************")
            print("lr: %.7f, reg: %.1f" %(lr, reg))
            model = SoftmaxClassifier()
            model.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=iters, verbose=False)

            # compute accuracy
            train_accuracy = compute_accuracy(y_train, model.predict(X_train))
            val_accuracy = compute_accuracy(y_val, model.predict(X_val))
            #print ('train accuracy: %.4f' %train_accuracy)
            #print ('validation accuracy: %.4f' %val_accuracy)

            # store accuracy in dictionary
            results[(lr, reg)] = (train_accuracy, val_accuracy)
            all_classifiers.append(model)

            # check if validation accuracy is best
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_softmax = model

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################       
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
