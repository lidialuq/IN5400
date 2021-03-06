

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import time
import os
import numpy as np
import PIL.Image
import sklearn.metrics
from RainforestDataset import RainforestDataset, ChannelSelect
from network import SingleNetwork, TwoNetworks
from datetime import datetime

# set seed for reproducibility
torch.manual_seed(0)

#-----------------------------------------------------------------------------
#-------------------USER DEFINED PARAMETERS-----------------------------------
#-----------------------------------------------------------------------------
MODEL = 'double4' #Choose between single3, single4 or double4
DATA_DIR = '/itf-fi-ml/shared/IN5400/2022_mandatory1'
#----------------------------------------------------------------------------


#'/mnt/CRAI-NAS/all/lidfer/IN5400/rainforest_mini'
print(f'Using model: {MODEL}')

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
RESULTS_PATH = os.path.join(CURRENT_PATH, 'results_'+MODEL)

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)


def train_epoch(model, trainloader, criterion, device, optimizer):

    model.train()
 
    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)
    
        # calculate the loss from minibatch.
        if MODEL == 'single3' or MODEL == 'single4':
            inputs = data['image'].to(device)        
            outputs = model(inputs)
        elif MODEL == 'double4':
            input1 =  data['image'][:,[0,1,2], : , :].to(device)    
            input2 = data['image'][:,[3], : , :].repeat(1,3,1,1).to(device)    
            outputs = model(input1, input2)

        labels = data['label'].to(device)

        # Calculate mean label-wise batch loss 
        optimizer.zero_grad()
        loss = criterion(outputs, labels.to(device).float()) #BCEWithLogitsLoss requires float
        losses.append(loss.item()) 
        loss.backward()
        optimizer.step()
      
    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    model.eval()
    
    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
       
          if MODEL == 'single3' or MODEL == 'single4':
              inputs = data['image'].to(device)        
              outputs = model(inputs)
          if MODEL == 'double4':
              input1 =  data['image'][:,[0,1,2], : , :].to(device)    
              input2 = data['image'][:,[3], : , :].repeat(1,3,1,1).to(device)    
              outputs = model(input1, input2)
              
          labels = data['label']

          loss = criterion(outputs, labels.to(device).float())
          losses.append(loss.item())
          
          sigmoid = nn.Sigmoid()
          outputs_sig = sigmoid(outputs)
          # collect prediction scores
          pred_array = outputs_sig.to('cpu').detach().numpy() # output of shape (batch_size, nr_labels)
          concat_pred = np.append(concat_pred, pred_array, axis=0)
          # collect labels
          labels = labels.to('cpu').detach().numpy()
          concat_labels = np.append(concat_labels, labels, axis=0)
          # collect filenames
          fnames.extend(data['filename'])
          

    avgprecs = sklearn.metrics.average_precision_score(concat_labels, concat_pred, 
                                                       average=None, pos_label=1)
      
    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval(dataloader_train, dataloader_test, model, criterion, optimizer, 
              scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss=train_epoch(model, dataloader_train, criterion, device, optimizer)
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss, concat_labels, concat_pred, fnames = \
        evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)
    
    dic = {'trainlosses': trainlosses,
           'testlosses': testlosses,
           'testperfs': testperfs,
           'best_epoch': best_epoch}
    
    with open(os.path.join(RESULTS_PATH, "metrics.pth"), 'wb') as f: 
        pickle.dump(dic, f)
        
    if avgperfmeasure > best_measure: #higher is better 
        bestweights = model.state_dict()
        best_epoch = epoch
        best_measure = avgperfmeasure
        torch.save(model.state_dict(), os.path.join(RESULTS_PATH, "model.pth"))


  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs

def set_up(): 
    config = dict()
    config['use_gpu'] = True
    config['gpu_nr'] = 0
    config['lr'] = 0.005
    config['batchsize_train'] = 32
    config['batchsize_val'] = 64
    config['maxnumepochs'] = 35
    config['scheduler_stepsize'] = 10
    config['scheduler_factor'] = 0.3
      
    # This is a dataset property.
    config['numcl'] = 17
    
    if MODEL == 'single3': 
        channels = [0,1,2]
    else:
        channels = [0,1,2,3]
        
    normalize_mean = [0.7476, 0.6534, 0.4757, 0.0960]
    normalize_std = [0.1677, 0.1828, 0.2137, 0.0284]
      
    # Data augmentations.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
            ChannelSelect(channels),
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
            ChannelSelect(channels),
        ]),
    }
      
      
    # Datasets
    image_datasets={}
    image_datasets['train']=RainforestDataset(root_dir=DATA_DIR, train=True, 
                                              transform=data_transforms['train'])
    image_datasets['val']=RainforestDataset(root_dir=DATA_DIR, train=False, 
                                            transform=data_transforms['val'])
      
    #print(image_datasets['train'][0]['image'].shape)
    # Dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(image_datasets['train'], 
                                      batch_size = config['batchsize_train'],
                                      num_workers=1)
    dataloaders['val'] = DataLoader(image_datasets['val'], 
                                      batch_size = config['batchsize_val'],
                                      num_workers=1)
      
    # Device
    if config['use_gpu'] == True:
        device = torch.device(f"cuda:{config['gpu_nr']}")
    else:
        device = torch.device('cpu')
      
    # Model
    if MODEL == 'single3': 
        model = SingleNetwork(pretrained_net = models.resnet18(pretrained = True))
    elif MODEL == 'single4':
        model = SingleNetwork(pretrained_net = models.resnet18(pretrained = True), 
                              weight_init="kaiminghe")
    elif MODEL == 'double4':
        pretrained_net1 = models.resnet18(pretrained = True)
        pretrained_net2 = models.resnet18(pretrained = True)
        model = TwoNetworks(pretrained_net1, pretrained_net2)
        
    model = model.to(device)
      
    lossfct = nn.BCEWithLogitsLoss(reduction = 'mean')
    
    # DONE? Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), 
                               lr = config['lr'],
                               )
      
    # Decay LR by a factor of 0.3 every 10 epochs
    scheduler = lr_scheduler.StepLR(optimizer,
                              step_size = config['scheduler_stepsize'],
                              gamma = config['scheduler_factor'],
                              )
    
    return dataloaders, model, lossfct, optimizer, scheduler, config['maxnumepochs'], device, config['numcl'] 
    

if __name__=='__main__':
    
    start = datetime.now()
    
    # Set up of model, dataloades and all hyperparameters
    dataloaders, model, lossfct, optimizer, scheduler, epochs, device, numcl = \
        set_up()
    # Train model, evaluate at each epoch
    best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = \
        traineval(dataloaders['train'], dataloaders['val'], model, 
                 lossfct, optimizer, scheduler, num_epochs = epochs, 
                 device = device , numcl = numcl)
        
    print(f'Best epoch: {best_epoch}')
    print(datetime.now() - start)


