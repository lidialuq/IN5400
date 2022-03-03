#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:59:50 2022

@author: lidia
"""
import torch
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from train import evaluate_meanavgprecision, set_up

MODEL = 'single3' #Choose between single3, single4 or double4

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(CURRENT_PATH, 'results_'+MODEL, 'model.pth')
print(f'Infering using model: {MODEL}')

with open(os.path.join(CURRENT_PATH, 'metrics.pth'), "rb") as f: 
    metrics = pickle.load(f)
    
plt.plot(metrics['trainlosses'])
plt.plot(metrics['testlosses'])
lst = [item[0] for item in metrics['testperfs']]
plt.plot(lst)


def tail_accuracy(t, labels, predictions):
    '''
    labels: lst of all samples of lists of all lables
    predictions: same
    '''
    
    tail_accuracy = np.zeros(17)
    
    for c in range(17):
        pred_larger_t = 0
        inside_sum = 0
        for label, pred in zip(labels, predictions):
            if pred[c] > t:
                pred_larger_t += 1
                if label[c]:
                    inside_sum  += 1
        
        tail_accuracy[c] = (1/pred_larger_t)*inside_sum
    return np.mean(tail_accuracy)
                    

# set up and infer on validation dataset
dataloaders, model, loss, optimizer, scheduler, epochs, device, numcl = \
    set_up()
    
# Load trained model TODO CHECK IF THIS WORKS
model.load_state_dict( torch.load(MODEL_PATH,  map_location=device))

avgprecs, mean_losses, labels, predictions, fnames = \
    evaluate_meanavgprecision(model, dataloaders['val'], loss, device, numcl)    

# calculate tail accuracies
t_list = np.linspace(0.5,1,num=10, endpoint=False)
tail_accuracies = []
for t in t_list: 
    mean_ta = tail_accuracy(t, labels, predictions)
    tail_accuracies.append(mean_ta)
    
# plot top and bottom 10 images
