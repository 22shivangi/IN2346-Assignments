import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=4)
        init.xavier_normal_(self.conv1.weight.data)
        #self.conv1.apply(self.init_weights)
        self.elu1 = nn.ELU()
        self.max1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        init.xavier_normal_(self.conv2.weight.data)
        self.elu2 = nn.ELU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4)
        init.xavier_normal_(self.conv3.weight.data)
        self.elu3 = nn.ELU()
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.3)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)
        init.xavier_normal_(self.conv4.weight.data)
        self.elu4 = nn.ELU()
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(p=0.4)

        self.fc1 = nn.Linear(in_features=2304,out_features=1000)
        init.xavier_uniform_(self.fc1.weight.data)
        self.dropout5 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=1000,out_features=1000)
        init.xavier_uniform_(self.fc2.weight.data)
        self.dropout6 = nn.Dropout2d(p=0.6)

        self.fc3 = nn.Linear(in_features=1000,out_features=30)
        init.xavier_uniform_(self.fc3.weight.data)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.max1(x)
        x = self.dropout1(x)
        #print(x.shape)

        x = self.conv2(x)
        x = self.elu2(x)
        x = self.max2(x)
        x = self.dropout2(x)
        #print(x.shape)

        x = self.conv3(x)
        x = self.elu3(x)
        x = self.max3(x)
        x = self.dropout3(x)
        #print(x.shape)

        x = self.conv4(x)
        x = self.elu4(x)
        x = self.max4(x)
        x = self.dropout4(x)


        #print(x.shape)
        x = x.view(-1,2304)

        x = self.fc1(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        x = self.dropout6(x)

        x = self.fc3(x)
       
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def init_weights(self,m):
        if isinstance(m,nn.Conv2d):
            print("Conv")
            init.normal_(m)
        elif isinstance(m,nn.Linear):
            print("FC")
            init.xavier_uniform_(m)

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


#model = KeypointModel()
#model.apply(model.init_weights)
