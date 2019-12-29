#push-relabel algorithm
import numpy as np
import time
import threading

def MaxFlow(C, s, t):
    n = len(C) # C is the capacity matrix
    F = [[0] * n for i in range(n)]

    # the residual capacity from u to v is C[u][v] - F[u][v]
    height = [0] * n # height of node
    excess = [0] * n # flow into node minus flow from node
    seen   = [0] * n # neighbours seen since last relabel
    # node "queue"
    nodelist = [i for i in range(n) if i != s and i != t]

    #push operation
    def push(u, v):
        send = min(excess[u], C[u][v] - F[u][v])
        F[u][v] += send
        F[v][u] -= send
        excess[u] -= send
        excess[v] += send


    #relabel operation
    def relabel(u):
        # find smallest new height making a push possible,
        # if such a push is possible at all
        min_height = float('inf')
        for v in range(n):
            if C[u][v] - F[u][v] > 0:
                min_height = min(min_height, height[v])
                height[u] = min_height + 1

    def discharge(u):
        while excess[u] > 0:
            if seen[u] < n: # check next neighbour
                v = seen[u]
                if C[u][v] - F[u][v] > 0 and height[u] > height[v]:
                    push(u, v)
                else:
                    seen[u] += 1
            else: # we have checked all neighbours. must relabel
                relabel(u)
                seen[u] = 0

    height[s] = n   # longest path from source to sink is less than n long
    excess[s] = float("inf") # send as much flow as possible to neighbours of source
    
    for v in range(n):
        push(s, v)

    def worker(a, thread_num):    
        p = a
        while p < len(nodelist):
            u = nodelist[p]
            old_height = height[u]
            discharge(u)
            if height[u] > old_height:
                nodelist.insert(a, nodelist.pop(p)) # move to front of list
                p = a # start from front of list
            else:
                p += thread_num

    threads = []
    thread_num = 5      #number of threads
    for i in range(thread_num):
        t = threading.Thread(target=worker, args=(i, thread_num))
        threads.append(t)
        t.start()

    for thread in threads:
	    thread.join()

    return sum(F[s])

# make a capacity graph
# node s  o  p  q  r  t
a = np.ones([902,902])
#print(a)
C = [[ 0, 3, 3, 0, 0, 0 ],  # s
     [ 0, 0, 2, 3, 0, 0 ],  # o
     [ 0, 0, 0, 0, 2, 0 ],  # p
     [ 0, 0, 0, 0, 4, 2 ],  # q
     [ 0, 0, 0, 0, 0, 2 ],  # r
     [ 0, 0, 0, 0, 0, 0 ]]  # t

source = 0     # A
sink = 901     # F
s = time.perf_counter()
max_flow_value = MaxFlow(a, source, sink)
e = time.perf_counter()
print ("Push-Relabeled(Preflow-push) algorithm")
print ("max_flow_value is: ", max_flow_value)
print ("runtime is: ", e-s)
