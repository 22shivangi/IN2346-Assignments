from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(ignore_index = -1)):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        for epoch in range(num_epochs):
            # TRAINING
            print("Epoch %d of %d" % (epoch + 1,num_epochs))
            model.train()
            train_losses=[]
            total = 0
            correct = 0
            for i, (inputs, targets) in enumerate(train_loader):
                targets = targets.type(torch.LongTensor)
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                train_losses.append(loss.data.cpu().numpy())
                self.train_loss_history.append(loss.data.cpu().numpy())
                train_loss = np.mean(train_losses)

                _, preds = outputs.max(1)

                #print(outputs)
                #print(preds)

                total += targets.size(0)
                correct += preds.eq(targets).sum().item()

                if i%log_nth==0 :
                    print('[Iteration %d of %d] Training loss: %.3f ' % (i + 1, iter_per_epoch,train_loss))

            acc = (correct/total)*100

            self.train_acc_history.append(acc)
            train_loss = np.mean(train_losses)
            print('TRAINING   acc = %.3f (%d/%d)   loss = %.3f  ' % (acc,correct,total, train_loss))




            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for i,(inputs, targets) in enumerate(val_loader):
                targets = targets.type(torch.LongTensor)
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss.data.cpu().numpy())

                _, preds = outputs.max(1)

                scores = np.mean((preds == targets).data.cpu().numpy())*100
                val_scores.append(scores)





            val_acc,val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            print('VALIDATION   acc = %.3f      loss = %.3f' % ( val_acc, val_loss))


        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
