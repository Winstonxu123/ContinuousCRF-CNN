#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:42:52 2017

@author: xu
"""

import caffe
import numpy as np


class scaleVariantLossLayer(caffe.layers):
    
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute!")
            
    def reshape(self,bottom,up):
        if bottom[0].count != bottom[1].count:
            raise Exception("two input must have the same dimension!")
        
        self.diff=np.zeros_like(bottom[0].data,dtype=np.float32)
        top[0].reshape(1)
        
     
    def forward(self,bottom,up):
        self.diff[...]=bottom[0].data-bottom[1].data
        top[0].data[...]=(np.sum(self.diff**2)/bottom[0].num)-0.5*(np.sum(self.diff)**2)/(bottom[0].num)**2
        
        
    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
        if i==0:
            sign=1
        else:
            sign=-1
        
        bottom[0].diff[...]=sign*