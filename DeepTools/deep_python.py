import numpy as np
from os.path import dirname, abspath
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import PIL
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
import glob
import re
import os
import math
import sys
from scipy import misc
import numpy
import pickle
import torch
import time
from PIL import Image
import subprocess
import scipy.io as sio
import models
import models_leaves
from threading import Thread, Lock
import DGC
import logging

def show(image_data, name):

        assert image_data is not None

        if type(image_data) is not np.ndarray:
            if len(image_data.shape) == 3:
                image_data = image_data.permute(2,0,1).numpy()

        elif type(image_data) is np.ndarray:
            if len(image_data.shape) == 2:
                image_data = image_data[:,::-1]
            #image_data = np.swapaxes(image_data, 0, 2)

        plt.imsave(name, image_data[...,::-1])


proper_polylines = {'fg':[],'bg':[]}
def process(rois, name):
	proper_polylines[name] = []
	for roi in rois:
		proper_polylines[name].append(DGC.polyline(roi))

[rois,segment,network, quality,deepsegment,mex,token] = pickle.load(open('data.p','rb'))
process(rois['fg'],'fg')
process(rois['bg'], 'bg')

img_raw = imread('./temp.tif')

oshape = img_raw.shape
w_original, h_original = oshape[1], oshape[0]

max_w = (2048*(int(quality)))/20
max_h = (2048*(int(quality)))/20

hh , ww = h_original, w_original
if h_original > max_h:
    hh = max_h

if w_original > max_w:
    ww = max_w

w, h = int(ww), int(hh)
img = imresize(img_raw, (h,w))

t_scale = (float(w)/oshape[1],float(h)/oshape[0])

img3 = None 


def segmentation_processing():
    	global img3, proper_polylines, w, h

	GC = DGC.GraphCutter()
	my_options = {'file': './temp.tif', 'image': None, 'resize': True, 'w': w, 'h': h, 'blur': True,
		      'show_figures': True, 'debug': True, 'log': True}
	my_parameters = {'bins': 8, 'sigma': 7.0, 'interaction_cost': 50, 'stroke_dt': True, 'hard_seeds': True, 'stroke_var': 50}
	my_polylines = proper_polylines

	out = GC.graph(polylines=my_polylines, options=my_options, parameters=my_parameters)
    	
	seg_image, contours_proper = out['segmentation'], out['contours']

	I = seg_image
	show(I,"./segmentation_result.png")
	myimg2 = (((I - I.min()) / (I.max() - I.min())) * 1.0).astype(np.uint8)
	myimg2 = np.expand_dims(myimg2,axis=2)

	myimg = np.array(img)
	myimg = np.multiply(myimg2,myimg)
	black = myimg2[:,:,0] == 0

	myimg[black] = [0.549*255, 0.570*255, 0.469*255]
	img3 = myimg.astype(np.uint8)
	
	show(img3,"./segmentation_result_masked.png")

	t_scale_inverse = (oshape[0]/float(h),oshape[1]/float(w)) # the sizes are inverted
	pickle.dump([contours_proper, t_scale_inverse],open("./contours.pkl","wb"))

if segment == "True":
	segmentation_processing()


if network == "Simple classification":

	dualnet = models.resnet50(pretrained=False)
	dualnet.fc = nn.Linear(2048, 5)
	criterion = nn.NLLLoss()
	m = nn.LogSoftmax()
	# cpu model for inference. The model itself is a gpu model, so we need to have the map_location arguments
	dualnet.load_state_dict(torch.load("/home/dimitris/Bisque/modules/PlanteomeDeepSegment/DeepModels/resnet_model.pth", map_location=lambda storage, loc:storage))
	dualnet.eval()

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.549,0.570,0.469),(0.008,0.0085,0.0135))])
	
	if deepsegment == "True" and segment == "True":
	    img_raw = img3

	img = imresize(img_raw, (int(224),int(224)))
	img_tensor = transform(img)
	img_tensor = img_tensor.expand(1,3,224,224)
	data = Variable(img_tensor, volatile=True)

	output_class = dualnet(data)
	softmax = nn.Softmax()
	output_class_s = softmax(output_class)
	pred = output_class_s.data.max(1)[1]
	conf = output_class_s.data.max(1)[0]
	print(pred)
	print(conf)

	# These go into the results.txt file and will be read from the main file when ready
	print "PREDICTION_C:",pred.numpy()[0]
	print "CONFIDENCE_C:","%.2f" % round(conf.numpy()[0],2)
	# print "DATA INFORMATION",rois[0],rois[1],rois[2],segment,network,mex,token

if network == "Leaf classification":

	dualnet = models_leaves.resnet18_leaf(pretrained=False)
	dualnet.reform()
	
	criterion = nn.NLLLoss()
	m = nn.LogSoftmax()
	# cpu model for inference. The model itself is a gpu model, so we need to have the map_location arguments
	dualnet.load_state_dict(torch.load("/home/dimitris/Bisque/modules/PlanteomeDeepSegment/DeepModels/leaf_model.pth", map_location=lambda storage, loc:storage))
	dualnet.eval()

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.7446,0.7655,0.7067),(0.277,0.24386,0.33867))])
	
	if deepsegment == "True" and segment == "True":
	    img_raw = img3

	img = imresize(img_raw, (int(224),int(224)))
	img_tensor = transform(img)
	img_tensor = img_tensor.expand(1,3,224,224)
	data = Variable(img_tensor, volatile=True)

	output_class = dualnet(data)	
	softmax = nn.Softmax()
	for i in range(6):
		print(softmax(output_class[i]).data.max(1)[1].numpy()[0])

