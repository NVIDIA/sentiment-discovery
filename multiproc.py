import torch
import sys
import os
import subprocess

argslist = list(sys.argv)[1:]

LOGDIR = 'distributed_logs'
if '--save' in argslist:
    savepath = os.path.splitext(os.path.basename(argslist[argslist.index('--save')+1]))[0]
else:
    savepath = 'model'
LOGDIR = os.path.join(LOGDIR, savepath)
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

world_size = torch.cuda.device_count()
if '--world_size' in argslist:
    argslist[argslist.index('--world_size')+1] = str(world_size)
else:
    argslist.append('--world_size')
    argslist.append(str(world_size))

for i in range(world_size):
    if '--rank' in argslist:
        argslist[argslist.index('--rank')+1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))
    #stdout = open(os.path.join(LOGDIR, str(i)+".log"), "w")
    stdout = None if i == 0 else open(os.path.join(LOGDIR, str(i)+".log"), "w")
    call = subprocess.Popen
    if i == world_size-1:
        call = subprocess.call
    call([str(sys.executable)]+argslist, stdout=stdout)


