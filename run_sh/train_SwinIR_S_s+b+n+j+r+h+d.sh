cd /home/notebook/data/personal/S9053103/MiOIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s_10w.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s+b_10w.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s+b+n_10w.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s+b+n+j_10w.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s+b+n+j+r_10w.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s+b+n+j+r+h_10w.yml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/SwinIR/MiO_002_SwinIR_S_s+b+n+j+r+h+d_10w_100w.yml
