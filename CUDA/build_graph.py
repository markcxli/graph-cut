#=== EC527 Final Project
#=== Author: Chenxi Li


import numpy as np
from numba import roc
from numba import cuda
from time import sleep
from math import *
import os

os.environ["NUMBA_CUDA_DRIVER"]="/share/pkg.7/cuda/10.1/install/lib64/libcudart.so"


@cuda.jit(device=True)
def add_neighborhood_edges(img, WEIGHTS_UP, WEIGHTS_DOWN, WEIGHTS_LEFT, WEIGHTS_RIGHT,x,y):
	#146.5-146.6
	k=146
	#foreground fades out if k goes down
	s=37
	#s=3
	# s=37.4 - 37.5
	# background cuts get more sensitive/sharp if s is small 
	#  Note: I flipped k and s, resulted graph is the same 
	h =len(img)
	w =len(img[0])
	if x+1 < w:
		diff = img[y, x]-img[y, x+1]
		w = k*exp(-(abs(diff**2)/s))
		WEIGHTS_RIGHT[y, x]=w
	if x-1 >= 0:
		diff2 = img[y, x]-img[y, x-1]
		w2 = k*exp(-(abs(diff**2)/s))
		WEIGHTS_LEFT[y, x-1]=w2
	if y-1 >= 0:
		diff3 = img[y, x]-img[y-1, x]
		w3 = k*exp(-(abs(diff**2)/s))
		WEIGHTS_UP[y-1, x]=w3
	if y+1 < h:
		diff4 = img[y, x]-img[y+1, x]
		w4 = k*exp(-(abs(diff**2)/s))
		WEIGHTS_DOWN[y+1, x]=w4

@cuda.jit(device=True)
def add_edge_test(img, WEIGHTS_UP, WEIGHTS_DOWN, WEIGHTS_LEFT, WEIGHTS_RIGHT, x,y ):
	h=len(img)
	w=len(img[0])
	if x+1 < w:
		WEIGHTS_RIGHT[y, x]=3
	if x-1 >=0:
		WEIGHTS_LEFT[y, x-1]=3
	if y-1 >=0:
		WEIGHTS_UP[y-1, x]=3
	if y+1 < h:
		WEIGHTS_DOWN[y,x]=3



@cuda.jit(device=True)
def initialize_excess_flow(img, PF, PB, EXCESS_FLOW,x,y):
	EXCESS_FLOW[y, x] = PF[y, x]*13-PB[y, x]*15
	
"""
@cuda.jit(device=True)
def initialize_sink_weights(PB,PF,SINK_WEIGHTS,x,y)
	SINK_WEIGHTS=PB[y,x]*15-PF[y,x]*13
"""

@cuda.jit
def make_graph(img, WEIGHTS_UP, WEIGHTS_DOWN, WEIGHTS_LEFT, WEIGHTS_RIGHT, EXCESS_FLOW, PF, PB):
	x = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.y
	y = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.x
	add_neighborhood_edges(img, WEIGHTS_UP,WEIGHTS_DOWN,WEIGHTS_LEFT,WEIGHTS_RIGHT,x,y)
	initialize_excess_flow(img, PF, PB, EXCESS_FLOW, x,y)
 	

def build_graph(gs,bs,img,wsu,wsd,wsl,wsr,ef,pf,pb):
	cuda.select_device(0)
	make_graph[(gs,gs),(bs,bs)](img,wsu,wsd,wsl,wsr,ef,pf,pb)
	return img,wsu,wsd,wsl,wsr,ef,pf,pb

