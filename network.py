import torch
import torch.nn as nn
from RainforestDataset import get_classes_list


class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        # select all parts of the two pretrained networks, except for
        # the last linear layer.
        self.fully_conv1 = nn.Sequential(*(list(pretrained_net1.children())[0:9]))
        self.fully_conv2 = nn.Sequential(*(list(pretrained_net2.children())[0:9]))
        # TODO create a linear layer that has in_channels equal to
        # the number of in_features from both networks summed together.
        self.linear = nn.Linear(1024, num_classes)


    def forward(self, inputs1, inputs2):
        # TODO feed the inputs through the fully convolutional parts
        # of the two networks that you initialised above, and then
        # concatenate the features before the linear layer.
        # And return the result.
        
        features1 = self.fully_conv1(inputs1)
        features2 = self.fully_conv2(inputs2)
        features_all = torch.cat((features1, features2), 1)
        #features_all = torch.flatten(features_all, 1,3)
        # print(features1.shape)
        # print(features2.shape)
        # print(features_all.shape)
        features_all = features_all.view(features_all.size(0), -1)
        out = self.linear(features_all)

        return out


class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init=None):
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()


        if weight_init is not None:
            # TODO Here we want an additional channel in the weights tensor, specifically in the first
            # conv2d layer so that there are weights for the infrared channel in the input aswell.
            current_weights = pretrained_net.conv1.weight

            if weight_init == "kaiminghe":
                w = torch.empty(64,1,7,7)
                w_he = nn.init.kaiming_normal_(w)
                weights = torch.cat((current_weights, w_he), 1)
                
            # TODO Create a new conv2d layer, and set the weights to be
            # what you created above. You will need to pass the weights to
            # torch.nn.Parameter() so that the weights are considered
            # a model parameter.
            # eg. first_conv_layer.weight = torch.nn.Parameter(your_new_weights)
            pretrained_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), 
                                   padding=(3,3), bias=False)
            #with torch.no_grad():
            #pretrained_net.conv1.weight.data.fill_(weights)
            pretrained_net.conv1.weight = torch.nn.Parameter(weights)
            

        # DONE? Overwrite the last linear layer.
        pretrained_net.fc = nn.Linear(512, num_classes)


        self.net = pretrained_net

    def forward(self, inputs):
        return self.net(inputs)





