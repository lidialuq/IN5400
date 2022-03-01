#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:59:50 2022

@author: lidia
"""

import pickle
import os
import matplotlib.pyplot as plt

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(CURRENT_PATH, 'metrics.pth'), "rb") as f: 
    metrics = pickle.load(f)
    
plt.plot(metrics['trainlosses'])
plt.plot(metrics['testlosses'])
lst = [item[0] for item in metrics['testperfs']]
plt.plot(lst)

