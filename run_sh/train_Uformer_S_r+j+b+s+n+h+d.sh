cd /home/notebook/data/personal/S9053103/MiOIR
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r_10w.yml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r+j_10w.yml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r+j+b_10w.yml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r+j+b+s_10w.yml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r+j+b+s+n_10w.yml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r+j+b+s+n+h_10w.yml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt /home/notebook/data/personal/S9053103/MiOIR/options/train/Uformer/MiO_004_Uformer_S_r+j+b+s+n+h+d_10w_100w.yml
