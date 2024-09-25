#%% imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
theta = np.load("params_20240426_140730_2.npy")
t = np.load("time_data_20240426_140730_2.npy")
v = np.load("voltage_data_20240426_140730_2.npy")

#%% break up signal into blocks
nRepeats = 5 # signal is repeated 5 times
nBlocks = 4 # chirp, charge, rest, HPCC (may need to break HPCC into charge, rest, discharge)
d = theta[-8:] # d1, d2, ..., d8 as described in separate file
vAll = []
tAll = []

t0 = 0 # real time
i0 = 0 # index number

for repeat in range(nRepeats):
    for block in range(nBlocks):
        if block==0: 
            # first block is always a chirp, duration = d8 (apparently twice that the first time haha!)
            if repeat==0:
                tEnd = t0 + 2*d[7]
            else:
                tEnd = t0 + d[7]
            iEnd = np.min(np.where(t>tEnd))-1
            vAll.append(v[i0:iEnd])
            tAll.append(t[i0:iEnd])
            print(i0, iEnd)
            i0 = iEnd
            t0 = t[iEnd]
        if block==1:
            # second block is always a charging step, duration = d2 (or when v=4.2!!)
            tEnd = t0 + d[1] + 200.
            iEnd = np.min(np.where(t>tEnd))-1
            if (v[i0:iEnd]>4.2).any():
                dur = np.min(np.where(v[t0:tEnd]>4.2)) - t0 # find the spot where it reaches 4.2!!
                iEnd = i0 + dur
                tEnd = t[iEnd]
                print("hit 4.2!")
            vAll.append(v[i0:iEnd])
            tAll.append(t[i0:iEnd])
            print(i0, iEnd)
            i0 = iEnd
            t0 = t[iEnd]
        if block==2:
            # third block is rest before HPCC and after charging, duration = d3
            tEnd = t0 + d[2]
            iEnd = np.min(np.where(t>tEnd))-1
            vAll.append(v[i0:iEnd])
            tAll.append(t[i0:iEnd])
            print(i0, iEnd)
            i0 = iEnd
            t0 = t[iEnd]
        if block==3:
            # fourth block is always the HPCC step, duration = 2*p5 + p6
            tEnd = t0 + 2*d[4] + d[5] + 300.
            if np.where(t>tEnd)[0].shape[0] == 0:
                iEnd = t.shape[0]-1
            else:
                iEnd = np.min(np.where(t>tEnd))-1
            vAll.append(v[i0:iEnd])
            tAll.append(t[i0:iEnd])
            print(i0, iEnd)
            i0 = iEnd
            t0 = t[iEnd]
        print("t0 = " + str(tAll[block+repeat*4][0]) + ", tFin = " +str(tAll[block+repeat*4][-1]) + ", at block " +str(block+repeat*4+1) ) 

#%% visualizing blocks
totalBlocks = nRepeats*nBlocks
for block in range(totalBlocks):
    plt.figure()
    plt.plot(tAll[block], vAll[block])
    plt.xlabel("time (sec)")
    plt.ylabel("voltage (V)")
    plt.title("Block Number "+str(block+1))
    plt.show()
