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

#-----------------------------------------------------------------------------
#-------------------USER DEFINED PARAMETERS-----------------------------------
#-----------------------------------------------------------------------------
# NOTE: MODEL has to be the same as MODEL in train.py, otherwise script will not run
MODEL = 'double4' #Choose between single3, single4 or double4
DATA_DIR = '/itf-fi-ml/shared/IN5400/2022_mandatory1'
#----------------------------------------------------------------------------

torch.manual_seed(0)
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(CURRENT_PATH, 'final_results', 'results_'+MODEL, 'model.pth')
SAVEFIG_PATH = os.path.join(CURRENT_PATH, 'final_results') 
print(f'Infering using model: {MODEL}')
print(f'Saving results in: {SAVEFIG_PATH}')


def save_predictions():
    # set up and infer on validation dataset
    dataloaders, model, loss, optimizer, scheduler, epochs, device, numcl = \
        set_up()
        
    # Load trained model 
    model.load_state_dict( torch.load(MODEL_PATH,  map_location=device))
    
    avgprecs, mean_losses, labels, predictions, fnames = \
        evaluate_meanavgprecision(model, dataloaders['val'], loss, device, numcl)   
    
    dic = {'avgprecs': avgprecs,
           'mean_losses': mean_losses,
           'labels': labels,
           'predictions': predictions,
           'fnames': fnames}
    with open(os.path.join(SAVEFIG_PATH, 'predictions_double4.pth'),'wb') as f:
        pickle.dump(dic, f)
        
    return dic


def plot_tail_accuracies():
    
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
                    if label[c] == 1:
                        inside_sum  += 1
            #print(f'{c}, {t}, {pred_larger_t}, {inside_sum}')
            try:
                tail_accuracy[c] = (1/pred_larger_t)*inside_sum
            except ZeroDivisionError:
                tail_accuracy[c] = np.nan
        return np.nanmean(tail_accuracy)
    
    with open(os.path.join(SAVEFIG_PATH, 'predictions_double4.pth'), 'rb') as f:
        dic = pickle.load(f)
        
    #calculate tail accuracies
    t_list = np.linspace(0.5,1,num=20, endpoint=False)
    tail_accuracies = []
    for t in t_list: 
        mean_ta = tail_accuracy(t, dic['labels'], dic['predictions'])
        tail_accuracies.append(mean_ta)
    
    #plot mean tail accuracies
    plt.figure()
    plt.plot(t_list, tail_accuracies)
    plt.xlabel('threshold (t)')
    plt.ylabel('Mean tail accuracy')
    plt.title('Mean tail accuracies at thresholds')
    plt.savefig(os.path.join(SAVEFIG_PATH, 'tailacc_double4.png'))

        
def show_top_bottom_images(c=12):
    DATA_DIR = '/home/lidia/Downloads/planet/train-jpg'
    with open(os.path.join(SAVEFIG_PATH, 'predictions_double4.pth'), 'rb') as f:
        dic = pickle.load(f)
    
    pred_primary = dic['predictions'][:,c]
    idx = list(range(0,len(dic['fnames'])))
    array = np.column_stack((pred_primary, idx))
    sorted_array = array[array[:,0].argsort()]
    sorted_array = sorted_array[sorted_array[:,0]>=0.5]
    
    # plot worse images for 'primary' class
    plt.figure()
    fig, axes = plt.subplots(2,5) 
    for i, ax in enumerate(axes.ravel()):
        img_idx = sorted_array[i][1].astype(int)
        image_label = dic['fnames'][img_idx]
        #image_path = os.path.join(DATA_DIR, 'train-tif-v2', image_label+'.tif')
        image_path = os.path.join(DATA_DIR, image_label+'.jpg')
        img = plt.imread(image_path)[:,:,0:3]
        ax.imshow(img)
        ax.set_axis_off()
        l_primary = dic['labels'][img_idx][c].astype(int)
        if l_primary == 1: 
            ax.set_title(image_label[6:] + '; TP')
        else:
            ax.set_title(image_label[6:] + '; FP')
    plt.savefig(os.path.join(SAVEFIG_PATH, 'worst_predictions.png'))
    
    # plot best images for 'primary' class
    plt.figure()
    fig, axes = plt.subplots(2,5) 
    for i, ax in enumerate(axes.ravel()):
        img_idx = sorted_array[-i+1][1].astype(int)
        image_label = dic['fnames'][img_idx]
        #image_path = os.path.join(DATA_DIR, 'train-tif-v2', image_label+'.tif')
        image_path = os.path.join(DATA_DIR, image_label+'.jpg')
        img = plt.imread(image_path)[:,:,0:3]
        ax.imshow(img)
        ax.set_axis_off()
        l_primary = dic['labels'][img_idx][c].astype(int)
        if l_primary == 1: 
            ax.set_title(image_label[6:] + '; TP')
        else:
            ax.set_title(image_label[6:] + '; FP')
    plt.savefig(os.path.join(SAVEFIG_PATH, 'best_predictions.png'))
        

def test_train_loss_plot():
    with open(MODEL_PATH.replace('model', 'metrics'), 'rb') as f:
        dic = pickle.load(f)
    mean_ap = [np.mean(ap) for ap in dic['testperfs']]
    plt.figure()
    fig,ax = plt.subplots()
    ax.plot(dic['trainlosses'], label = 'Train loss')
    ax.plot(dic['testlosses'], label = 'Test loss')
    ax.set_xlabel('epoch number')
    ax.set_ylabel('loss')
    ax.set_title('Train and test loss')
    ax2=ax.twinx()
    ax2.plot(mean_ap, label = 'Mean AP', color='g')
    ax2.set_ylabel('mean average precision')
    fig.legend(loc="upper right", bbox_to_anchor=(1,0.9), bbox_transform=ax.transAxes)
    plt.savefig(os.path.join(SAVEFIG_PATH, 'loss_and_meanap.png'))
    

def reproduction_rutine(): 
    with open(os.path.join(SAVEFIG_PATH, 'predictions_double4.pth'), 'rb') as f:
        dic = pickle.load(f)
    
    # saved predictions for validation dataset
    saved_predictions = dic['predictions']
    
    # predictions for validation dataset computed on the fly
    dataloaders, model, loss, optimizer, scheduler, epochs, device, numcl = \
        set_up()
    model.load_state_dict( torch.load(MODEL_PATH,  map_location=device))
    avgprecs, mean_losses, labels, predictions, fnames = \
        evaluate_meanavgprecision(model, dataloaders['val'], loss, device, numcl)   
    
    # compute and output difference
    diff = predictions[0] - saved_predictions[0]
    added_diff = np.sum(diff)
    print(f'Difference between on-the-fly and saved predictions added over all validation samples and classes: {added_diff}')
    print('Note: very small but non-zero differences can occur when on-the-fly predictions are done on a GPU different to the one used when saving the predictions.')

    return diff



if __name__=='__main__':   
    #dictionary = save_predictions()
    #show_top_bottom_images()
    #test_train_loss_plot()
    #plot_tail_accuracies()
    diff = reproduction_rutine()
    


