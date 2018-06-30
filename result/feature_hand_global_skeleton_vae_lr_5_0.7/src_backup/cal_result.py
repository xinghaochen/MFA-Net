#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:31:57 2017

@author: chenxh
"""

import sys
import numpy as np

if len(sys.argv) < 2:
    print 'Usage: cal_result.py save_postfix'
    exit()
save_postfix = sys.argv[1]
save_dir = 'result/'+save_postfix
res = np.loadtxt(save_dir+'/resulst.txt')
print res
mean_res = np.mean(res[:,1])
print 'acc for {}: '.format(save_postfix), mean_res

    