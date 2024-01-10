import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel



load_net = torch.load('/home/notebook/data/personal/S9053103/CL_code/experiments/CLDF_028_PromptIR_sbnjrhd_100w/net_g_latest.pth', map_location=lambda storage, loc: storage)

# remove unnecessary 'module.'
save_dict=deepcopy(load_net)
for k, v in load_net['params'].items():
    new_k=k.replace('cond_net','F_ext_net')
    new_k=new_k.replace('cond_','prompt_')
    new_k=new_k.replace('cond_scale5','prompt_scale2')
    new_k=new_k.replace('cond_scale6','prompt_scale3')
    new_k=new_k.replace('cond_scale7','prompt_scale4')
    new_k=new_k.replace('cond_scale8','prompt_scale5')
    new_k=new_k.replace('cond_scaleout','prompt_scaleout')
    new_k=new_k.replace('cond_shift4','prompt_shift1')
    new_k=new_k.replace('cond_shift5','prompt_shift2')
    new_k=new_k.replace('cond_shift6','prompt_shift3')
    new_k=new_k.replace('cond_shift7','prompt_shift4')
    new_k=new_k.replace('cond_shift8','prompt_shift5')
    new_k=new_k.replace('cond_shiftout','prompt_shiftout')

    if 'cond_net' in k or 'cond_' in k or 'cond_shift' in k:
        print(k+"   to   "+new_k)
        save_dict['params'][new_k]=save_dict['params'].pop(k)




save_path='/home/notebook/data/personal/S9053103/MiOIR/pretrained_model/PromptIR_M.pth'
torch.save(save_dict, save_path)

load_net = torch.load(save_path, map_location=lambda storage, loc: storage)



print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
for k, v in load_net['params'].items():
    print(k)

