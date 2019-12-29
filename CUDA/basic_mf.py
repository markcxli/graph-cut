#=== EC527 Final Project
#=== Author: Chenxi Li



"""
4 WxH arrays for residual edge capacities
2 WxH array for heights (double buffering)
WxH array for excess flow
"""

# All edges which are from a reachable vertex to non-reachable vertex are minimum cut edges. Print all such edges.
import numpy as np
from numba import *
from time import sleep
from build_graph import *
from min_cut import *
import cv2
import sys
import time
from operator import *
#GRAPH_SIZE= 500
#img = np.random.randint(255, size=(GRAPH_SIZE, GRAPH_SIZE))
img = cv2.imread("rose.jpg", cv2.IMREAD_GRAYSCALE)
GRAPH_SIZE = len(img)
BLOCK_SIZE = 25
TILE_SIZE = BLOCK_SIZE
GRID_SIZE = GRAPH_SIZE // BLOCK_SIZE

PF=np.random.randint(10,size=(GRAPH_SIZE, GRAPH_SIZE))*0.1
PB=np.random.randint(10,size=(GRAPH_SIZE, GRAPH_SIZE))*0.1

WEIGHTS_UP   = np.zeros((GRAPH_SIZE,GRAPH_SIZE))
WEIGHTS_DOWN = np.zeros((GRAPH_SIZE,GRAPH_SIZE))
WEIGHTS_LEFT = np.zeros((GRAPH_SIZE,GRAPH_SIZE))
WEIGHTS_RIGHT= np.zeros((GRAPH_SIZE,GRAPH_SIZE))

#WEIGHTS_SINK = np.zeros((GRAPH_SIZE,GRAPH_SIZE))

HEIGHTS= np.zeros((GRAPH_SIZE,GRAPH_SIZE),dtype=np.int32)
HEIGHTS_BUFFER= np.zeros((GRAPH_SIZE, GRAPH_SIZE),dtype=np.uint32)

EXCESS_FLOW = np.zeros((GRAPH_SIZE,GRAPH_SIZE))

EXCESS_TILE = np.zeros((GRAPH_SIZE,GRAPH_SIZE))

ACTIVE_NODES = np.zeros((GRAPH_SIZE,GRAPH_SIZE), dtype=bool)
ACTIVE_TILES = np.ones((GRID_SIZE,GRID_SIZE), dtype=bool)
COMPR_CAP = np.zeros((GRAPH_SIZE, GRAPH_SIZE), dtype=np.uint8)
#FINAL OUTPUT
SEG = np.zeros((GRAPH_SIZE,GRAPH_SIZE))

"""
below kernels does push and relabel on each tile. host will 
will them untill there is no active node in the tile. let's assume
all tiles are squares
 __________
|->|  |   |
 __________
|->|  |   |
 __________
|->|  |   |
 __________
"""

"""
void push(x, excess_flow, capacity, const height)
	if active(x) do
		foreach y=neighbor(x)
		if height(y) == height(x) – 1 do // check height
			flow = min( capacity(x,y), excess_flow(x)); // pushed flow
			excess_flow(x) -= flow; excess_flow(y) += flow; // update excess flow
			capacity(x,y) -= flow; capacity(y,x) += flow; // update edge cap.
		done
	end
done
"""


"""
ef = 0; for k=0...M-1 ef += s_ef(k) flow = min(right(x+k),ef) right(x+k)-=flow;
		s_ef(k)=ef-flow;
	ef = flow;
end
"""

# give access to next node

@cuda.jit(device=True)
def isactive(height,excess):
	return (height < GRAPH_SIZE*GRAPH_SIZE) and (excess > 0)



@cuda.jit(device=True)
def right_wave(sH, sE, x, y, WEIGHTS_RIGHT, CC, EXCESS_FLOW, EXCESS_TILE, BORDER_H):
	offy=cuda.blockIdx.y*cuda.blockDim.x; offx=cuda.blockIdx.x*cuda.blockDim.x
	tile_len=cuda.blockDim.x; ef = 0; nei_h=0
	
	for i in range(0,tile_len): 
		if offx+i==GRAPH_SIZE-1: #hit right bound, can't push! 
			sE[y, i] = ef 
			break
		h=sH[y,i]
		e=sE[y,i]
		if isactive(h,e):
			#print("active, gonna push\n")
			#print("neigbor h for  ", offx+i," ", offy+y, " is ",sH[y,i+1])
			if i<tile_len-1:
				nei_h=sH[y,i+1]
			else:
				nei_h=BORDER_H
			#print("active, gonna push\n")
			#print("neigbor h for  ", offx+i," ", offy+y, " is ", nei_h)

			if h==nei_h+1:    #problem: out of bound
			#	print("good h, gonna push\n")
				w  = WEIGHTS_RIGHT[offy+y, offx+i]
				ef = ef + e
				flow = min(w, ef)
				cc = CC[offy+y,offx+i] 
				if (w>flow):
					CC[offy+y,offx+i]=cc|1
				else:
					CC[offy+y,offx+i]=cc&(~1)
				WEIGHTS_RIGHT[offy+y, offx+i] = w-flow
				sE[y, i] = ef - flow
				ef = flow
				
	if offx+tile_len < GRAPH_SIZE:
		EXCESS_TILE[offy+y][offx+tile_len] = ef
	


@cuda.jit(device=True)
def left_wave(sH, sE, x, y, WEIGHTS_LEFT, CC, EXCESS_FLOW, EXCESS_TILE, BORDER_H):   
	offy=cuda.blockIdx.y*cuda.blockDim.x; offx=cuda.blockIdx.x*cuda.blockDim.x
	tile_len = cuda.blockDim.x; ef = 0; nei_h=0
	for i in range(1,tile_len+1):
		if offx+ tile_len -i==0: #hit img left bound
			sE[y, tile_len -i] = ef
			break
		h=sH[y,tile_len-i]
		e=sE[y,tile_len-i]
		if isactive(h,e):
			if tile_len-i>0:
				nei_h=sH[y, tile_len-i-1]
			else:
				nei_h=BORDER_H
			if h==nei_h+1: 
				w = WEIGHTS_LEFT[offy+y, offx+tile_len-i]
				ef = ef + e
				flow = min(w, ef)
				cc=CC[offy+y, offx+tile_len-i]
				if (w>flow):
					CC[offy+y, offx+tile_len-i]=cc|4
				else:
					CC[offy+y, offx+tile_len-i]=cc&(~4)
				WEIGHTS_LEFT[offy+y, offx+tile_len-i] = w-flow
				sE[y, tile_len-i] = ef - flow
				ef = flow
	if (offx-1)>=0:  #hit image left bound
		EXCESS_TILE[offy+y][offx-1] = ef

@cuda.jit(device=True)
def up_wave(sH, sE, x, y, WEIGHTS_UP, CC, EXCESS_FLOW, EXCESS_TILE, BORDER_H):
    # note: assume square image, and because dim.y is 1, we use dim.x for y length
	offy=cuda.blockIdx.y*cuda.blockDim.x;offx=cuda.blockIdx.x*cuda.blockDim.x
	tile_len=cuda.blockDim.x;ef = 0;nei_h=0
	for i in range(1,tile_len+1):
		if offy+tile_len-i==0:   #hit img top, don't push
			sE[tile_len-i, x] = ef
			break
		h=sH[tile_len-i,x]
		e=sE[tile_len-i,x]
		if isactive(h,e):
			if tile_len-i>0:
				nei_h=sH[tile_len-i-1,x]
			else:
				nei_h=BORDER_H
			if  h==nei_h+1: 
				w=WEIGHTS_UP[offy+tile_len-i,offx+x]
				ef=ef+e
				flow=min(w,ef)
				cc=CC[offy+tile_len-i,offx+x]
				if (w>flow):
					CC[offy+tile_len-i,offx+x]=cc|8
				else:
					CC[offy+tile_len-i,offx+x]=cc&(~8)
				WEIGHTS_UP[offy+tile_len-i,offx+x] = w-flow
				sE[tile_len-i, x]=ef-flow
				ef=flow
	if (offy-1)>=0:
		EXCESS_TILE[offy-1][offx+x]=ef

@cuda.jit(device=True)
def down_wave(sH, sE, x, y, WEIGHTS_DOWN, CC, EXCESS_FLOW, EXCESS_TILE, BORDER_H):
	offy=cuda.blockIdx.y*cuda.blockDim.x;offx=cuda.blockIdx.x*cuda.blockDim.x
	tile_len=cuda.blockDim.x;ef = 0;nei_h=0
	for i in range(0,tile_len):
		if offy+i==GRAPH_SIZE-1: #hit img bottom, keep excess flow, don't push
			sE[i,x] = ef
			break
		h=sH[i,x]
		e=sE[i,x]
		if isactive(h,e):
			if i<tile_len-1:
				nei_h=sH[i+1,x]	
			else:
				nei_h=BORDER_H
			if h==nei_h+1:
				w=WEIGHTS_DOWN[offy+i,offx+x]
				ef=ef+e
				flow = min(w,ef)
				cc=CC[offy+i,offx+x]
				if (w>flow):
					CC[offy+i,offx+x]=cc|2
				else:
					CC[offy+i,offx+x]=cc&(~2)
				WEIGHTS_DOWN[offy+i,offx+x]=w-flow
				sE[i,x]=ef-flow
				ef=flow
	if offy+tile_len<GRAPH_SIZE:
		EXCESS_TILE[offy+tile_len][offx+x] = ef

#this will be updatedi, or deleted
@cuda.jit() # for this kernel, use (gs,gs), (bs,bs)
# before launching this kernel, copy HEIGHTS into HEIGHTS_BUFFER, and empty HEIGHTS for writing
def local_relabel(EXCESS_FLOW, CC, H, HB, ACTIVE_NODES):
	# this is a seperarte kernel, each thread relabes one node
	
	offx=cuda.blockIdx.x*cuda.blockDim.x; offy=cuda.blockIdx.y*cuda.blockDim.y
	x=cuda.threadIdx.x; y=cuda.threadIdx.y; tile_len=cuda.blockDim.x
	e=EXCESS_FLOW[offy+y,offx+x]
	h=HB[offy+y,offx+x]
	c = CC[offy+y,offx+x]
	if isactive(h,e):
		min_h=GRAPH_SIZE*GRAPH_SIZE
		if offx+x+1<GRAPH_SIZE:
			if c == (c|1): #right edge available
				min_h=min(min_h, HB[offy+y, offx+x+1]+1)
		#		print("right  h  for ", offx+x, " ", offy+y, " is ",HB[offy+y, offx+x])
		if offy+y-1>=0:
			if c == (c|8): #up edge available
				min_h=min(min_h, HB[offy+y-1,offx+x]+1)
		#		print("up   h  for ", offx+x, " ", offy+y, " is ",HB[offy+y, offx+x])
		if offy+y+1<GRAPH_SIZE:
			if c == (c|2): #down edge available
				min_h=min(min_h, HB[offy+y+1,offx+x]+1)
		#		print("down  h for ", offx+x, " ", offy+y, " is ",HB[offy+y, offx+x] )
		if offx+x-1>=0:
			if c == (c|4): #left edge available
				min_h=min(min_h, HB[offy+y, offx+x-1]+1)
		#		print("left  h for ", offx+x, " ", offy+y, " is ",HB[offy+y, offx+x])

		H[offy+y,offx+x]=min_h
		
	else:
		H[offy+y,offx+x]=h
	ACTIVE_NODES[offy+y, offx+x]=isactive(H[offy+y,offx+x],e)			
	
	
@cuda.jit(device=True)
def build_tile(sH, sE, HEIGHTS, EXCESS_FLOW, EXCESS_TILE,  y, TYPE):
	offx=cuda.blockIdx.x * cuda.blockDim.x; offy=cuda.blockIdx.y * cuda.blockDim.x
	tile_len=cuda.blockDim.x
	for x in range(tile_len):
		sE[y, x]  = EXCESS_FLOW[offy+y, offx+x]	
		sH[y, x]  = HEIGHTS[offy+y, offx+x]
	if TYPE==0: #after a right wave
		if offx>0: #left bound  
			sE[y, 0] = sE[y, 0] + EXCESS_TILE[offy+y,offx]
			EXCESS_TILE[offy+y,offx]=0
	if TYPE==2: #after a left wave
		if offx+tile_len-1<GRAPH_SIZE:  #right bound
			sE[y, tile_len-1] = sE[y, tile_len-1] + EXCESS_TILE[offy+y,offx+tile_len-1]
			EXCESS_TILE[offy+y,offx+tile_len-1]=0
	if TYPE==1: #aftaer down push
		if offy>0:  #upper bound
			sE[0,y] = sE[0, y] + EXCESS_TILE[offy, offx+y]
			EXCESS_TILE[offy,offx+y]=0
	if TYPE==3: #AFTER up push
		if offy+tile_len-1<GRAPH_SIZE: #lower bound
			sE[tile_len-1,y] = sE[tile_len-1, y] + EXCESS_TILE[offy+tile_len-1, offx+y]
			EXCESS_TILE[offy+tile_len-1,offx+y]=0


@cuda.jit()
def push_right(WEIGHTS,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,TYPE):
	
    # this kernel is only for pushing in one direction
	# need to sync up blocks in host after
	sH = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	sE = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	lx = cuda.threadIdx.x; tile_len = cuda.blockDim.x	
	bx = cuda.blockIdx.x; by = cuda.blockIdx.y
	offx = bx*cuda.blockDim.x;offy = by*cuda.blockDim.x
	BORDER_H=HEIGHTS[offy+lx,offx+tile_len]

	bt_type=3
	build_tile(sH, sE, HEIGHTS, EXCESS_FLOW, EXCESS_TILE, lx, bt_type)
	cuda.syncthreads()
	if ACTIVE_TILES[by,bx]:
		right_wave(sH, sE, 0, lx, WEIGHTS, COMPR_CAP, EXCESS_FLOW, EXCESS_TILE, BORDER_H)
		for i in range(tile_len):
			EXCESS_FLOW[offy+lx, offx+i]= sE[lx, i]
	
@cuda.jit()
def push_down(WEIGHTS,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,TYPE):
	
	sH = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	sE = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	lx = cuda.threadIdx.x; tile_len = cuda.blockDim.x	
	bx = cuda.blockIdx.x; by = cuda.blockIdx.y
	offx = bx*cuda.blockDim.x;offy = by*cuda.blockDim.x
	BORDER_H=HEIGHTS[offy+tile_len,offx+lx]

	bt_type=0
	build_tile(sH, sE, HEIGHTS, EXCESS_FLOW, EXCESS_TILE, lx, bt_type)
	cuda.syncthreads()
	if ACTIVE_TILES[by,bx]:
		down_wave(sH, sE, lx,0, WEIGHTS, COMPR_CAP, EXCESS_FLOW, EXCESS_TILE,BORDER_H)
		for i in range(tile_len):
			EXCESS_FLOW[offy+lx, offx+i]= sE[lx, i]
	
@cuda.jit()
def push_left(WEIGHTS,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,TYPE):
	
	sH = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	sE = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	lx = cuda.threadIdx.x; tile_len = cuda.blockDim.x	
	bx = cuda.blockIdx.x; by = cuda.blockIdx.y
	offx = bx*cuda.blockDim.x;offy = by*cuda.blockDim.x
	BORDER_H=HEIGHTS[offy+lx,offx-1]

	bt_type=1
	build_tile(sH, sE, HEIGHTS, EXCESS_FLOW, EXCESS_TILE, lx, bt_type)
	cuda.syncthreads()
	if ACTIVE_TILES[by,bx]:
		left_wave(sH, sE, 0,lx, WEIGHTS, COMPR_CAP, EXCESS_FLOW, EXCESS_TILE, BORDER_H)
		for i in range(tile_len):
			EXCESS_FLOW[offy+lx, offx+i]= sE[lx, i]
	
@cuda.jit()
def push_up(WEIGHTS,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,TYPE):
	
	sH = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	sE = cuda.shared.array(shape=(BLOCK_SIZE,BLOCK_SIZE), dtype=int32)
	lx = cuda.threadIdx.x; tile_len = cuda.blockDim.x
	bx = cuda.blockIdx.x; by = cuda.blockIdx.y
	offx = bx*cuda.blockDim.x;offy = by*cuda.blockDim.x
	BORDER_H=HEIGHTS[offy-1,offx+lx]

	bt_type=2
	build_tile(sH, sE, HEIGHTS, EXCESS_FLOW, EXCESS_TILE, lx, bt_type)
	cuda.syncthreads()
	if ACTIVE_TILES[by,bx]:
		up_wave(sH, sE, lx,0, WEIGHTS, COMPR_CAP, EXCESS_FLOW, EXCESS_TILE, BORDER_H)
		for i in range(tile_len):
			EXCESS_FLOW[offy+lx, offx+i]= sE[lx, i]
	

@cuda.jit()
def global_bfs_t(H,EXCESS_FLOW,PF,PB,ASSIGNED_MASK):
	
	offx = cuda.blockIdx.x*cuda.blockDim.x; offy = cuda.blockIdx.y*cuda.blockDim.x
	x = cuda.threadIdx.x; y = cuda.threadIdx.y
	h=H[offy+y,offx+x];e=EXCESS_FLOW[offy+y,offx+x]
	if EXCESS_FLOW[offy+y,offx+x]<0:
		H[offy+y,offx+x]=1
		ASSIGNED_MASK[offy+y,offx+x]=True
	else: 
		H[offy+y,offx+x]=GRAPH_SIZE*GRAPH_SIZE


#for k=1, all pixel nodes with non-zero residual capacity to t are assigned height of 1
#every unassigned node assigns itself a height of k+1 if any of its neighbors have a height of k
#update new heigh value in global height array
@cuda.jit() # need to copy H to HB before calling this kenel
def global_bfs(H, CC, iters, isover, ASSIGNED_MASK):
	
	offx = cuda.blockIdx.x*cuda.blockDim.x;offy = cuda.blockIdx.y*cuda.blockDim.x
	x = cuda.threadIdx.x; y = cuda.threadIdx.y
	c=CC[offy+y,offx+x]; 
	h = H[offy+y,offx+x]; #	if h<GRAPH_SIZE*GRAPH_SIZE and not ASSIGNED_MASK[offy+y,offx+x]:
	if not ASSIGNED_MASK[offy+y,offx+x]:
		if offx+x+1<GRAPH_SIZE:
			if (c==(c|1)) & H[offy+y,offx+x+1]==iters and ASSIGNED_MASK[offy+y,offx+x+1]:
				isover=True
				H[offy+y,offx+x]=iters+1 
				ASSIGNED_MASK[offy+y,offx+x]=1
		if offx+x-1>=0:
			if (c==(c|4)) & H[offy+y,offx+x-1]==iters and ASSIGNED_MASK[offy+y,offx+x-1]:
				isover=True
				H[offy+y,offx+x]=iters+1
				ASSIGNED_MASK[offy+y,offx+x]=1
		if offy+y+1<GRAPH_SIZE:
			if (c==(c|2)) & H[offy+y+1,offx+x]==iters and ASSIGNED_MASK[offy+y+1,offx+x]:
				isover=True
				H[offy+y,offx+x]=iters+1
				ASSIGNED_MASK[offy+y,offx+x]=1
		if offy+y-1>=0:
			if (c==(c|8)) & H[offy+y-1,offx+x]==iters and ASSIGNED_MASK[offy+y-1,offx+x+1]:
				isover=True
				H[offy+y,offx+x]=iters+1
				ASSIGNED_MASK[offy+y,offx+x]=1

	# fix: update ACTIVE_NODES here
	ACTIVE_NODES[offy+y,offx+x]=isactive(H[offy+y,offx+x])
	cuda.syncthreads()
	

	

@cuda.jit()
def reduce_active_nodes(ACTIVE_NODES, ACTIVE_TILES, L):
	
	offx = cuda.threadIdx.x*L
	offy = cuda.threadIdx.y*L
	x = cuda.threadIdx.x
	y = cuda.threadIdx.y
	for j in range(L):
		for i in range(L):
			if ACTIVE_NODES[offy+j,offx+i]==True:
				ACTIVE_TILES[y,x]=True
				return	
	ACTIVE_TILES[y,x]=False
	
@cuda.jit()
def initialize_compressed_capacities(WL,WR,WU,WD,CC):
	
	# set initilial compressed capacity for first relabelling
	x=cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
	y=cuda.blockDim.y*cuda.blockIdx.y+cuda.threadIdx.y
	R=0;D=0;L=0;U=0;
	R=1&(WR[y,x]>0)
	D=2&(WD[y,x]>0)
	L=4&(WL[y,x]>0)
	U=8&(WU[y,x]>0)
	CC[y,x] = (R | D | L | U)
	
def has_converged():
	#return np.sum(ACTIVE_NODES)==0
	return np.sum(ACTIVE_TILES)==0



def maxflow():
	cuda.select_device(0)
	threadsperblock = (BLOCK_SIZE,1)
	blockspergrid = (GRID_SIZE,GRID_SIZE)
	iters=0
	EXCESS_TILE=np.zeros((GRAPH_SIZE,GRAPH_SIZE))
	ASSIGNED_MASK=np.zeros((GRAPH_SIZE,GRAPH_SIZE),dtype=bool)

	initialize_compressed_capacities[blockspergrid,(BLOCK_SIZE,BLOCK_SIZE)](WEIGHTS_LEFT,WEIGHTS_RIGHT,WEIGHTS_UP,WEIGHTS_DOWN,COMPR_CAP)	

	start=cuda.event(timing=True)
	stop=cuda.event(timing=True)
	start.record()
	begin=time.time()
	
	
	while (not has_converged()) or iters<5:
	
		push_right[blockspergrid, (BLOCK_SIZE,1)](WEIGHTS_RIGHT,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,0)	
		cuda.synchronize()

		push_down[blockspergrid, (BLOCK_SIZE,1)](WEIGHTS_DOWN,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,1)
		cuda.synchronize()
	
		push_left[blockspergrid, (BLOCK_SIZE,1)](WEIGHTS_LEFT,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,2)
		cuda.synchronize()
		
		push_up[blockspergrid, (BLOCK_SIZE,1)](WEIGHTS_UP,EXCESS_FLOW,EXCESS_TILE,HEIGHTS,ACTIVE_TILES,ACTIVE_NODES,COMPR_CAP,3)
		cuda.synchronize()

		HEIGHTS_BUFFER=np.copy(HEIGHTS)
		HEIGHTS.fill(0)
		local_relabel[blockspergrid, (BLOCK_SIZE,BLOCK_SIZE)](EXCESS_FLOW, COMPR_CAP, HEIGHTS, HEIGHTS_BUFFER, ACTIVE_NODES)
		cuda.synchronize()

		if iters%6==0:
			isover=True
			ASSIGNED_MASK.fill(0)
			global_bfs_t[(GRID_SIZE,GRID_SIZE),(BLOCK_SIZE,BLOCK_SIZE)](HEIGHTS,EXCESS_FLOW,PF,PB,ASSIGNED_MASK)
			cuda.synchronize()
			gl_count=1
			while (isover):
				isover=False
				global_bfs[(GRID_SIZE,GRID_SIZE),(BLOCK_SIZE,BLOCK_SIZE)](HEIGHTS,COMPR_CAP,gl_count,isover,ASSIGNED_MASK)
				gl_count=gl_count+1
				cuda.synchronize()

		reduce_active_nodes[1,(GRID_SIZE, GRID_SIZE)](ACTIVE_NODES,ACTIVE_TILES,BLOCK_SIZE)
		cuda.synchronize()
	
		iters=iters+1


		print("This is iteration %d"%iters)
	end=time.time()
	print("-Done in "+str(end-begin))

	stop.record()
	stop.synchronize()
	GPU_TIME=cuda.event_elapsed_time(start,stop)

	print("It took %d iterations to converge"%iters)
	print("Total GPU time: %dms"%(GPU_TIME))


def generate_image_seg_mask(WL,WR,WU,WD,img):
	WR_c=np.where(WR>0,0,1)
	WL_c=np.where(WR>0,0,1)
	WU_c=np.where(WR>0,0,1)
	WD_c=np.where(WR>0,0,1)
	
	Mask1=WR_c+WL_c
	Mask2=WU_c+WD_c
	Mask=Mask1+Mask2
	Mask=np.where(Mask>0,1,0)
	Mask=np.multiply(Mask,255)   # use white color for segmentation boudry

	img=img+Mask
	img=np.where(img>255,255,img)
	cv2.imwrite('seg.png',img)



#test with small graph, wait, need to count active tiles
if __name__ == "__main__":

	img, WEIGHTS_UP, WEIGHTS_DOWN, WEIGHTS_LEFT, WEIGHTS_RIGHT, EXCESS_FLOW, PF, PB=build_graph(GRID_SIZE, BLOCK_SIZE,img,WEIGHTS_UP,WEIGHTS_DOWN,WEIGHTS_LEFT,WEIGHTS_RIGHT,EXCESS_FLOW,PF,PB)

	ORG_WU = np.copy(WEIGHTS_UP)
	ORG_WD = np.copy(WEIGHTS_DOWN)
	ORG_WL = np.copy(WEIGHTS_LEFT)
	ORG_WR = np.copy(WEIGHTS_RIGHT)
	
	maxflow()
	generate_image_seg_mask(WEIGHTS_LEFT,WEIGHTS_RIGHT,WEIGHTS_UP,WEIGHTS_DOWN,img)
	#SEG = mincut(WEIGHTS_UP, WEIGHTS_DOWN, WEIGHTS_LEFT, WEIGHTS_RIGHT, ORG_WU, ORG_WD, ORG_WL, ORG_WR, SEG, GRID_SIZE, BLOCK_SIZE)

	wl=0
	wr=0
	wu=0
	wd=0

	for j in range(len(img)):
		for i in range(len(img[0])):
			if ORG_WU[j,i]!=0 and WEIGHTS_UP[j,i]==0:
				wu=wu+1
			if ORG_WD[j,i]!=0 and WEIGHTS_DOWN[j,i]==0:
				wr=wr+1
			if ORG_WL[j,i]!=0 and WEIGHTS_LEFT[j,i]==0:
				wl=wl+1
			if ORG_WR[j,i]!=0 and WEIGHTS_RIGHT[j,i]==0:
				wd=wd+1
	
	print("num of WL changed: %d" % wl)
	print("num of WR changed: %d" % wr)
	print("num of WU changed: %d" % wu)
	print("num of WD changed: %d" % wd)

