# mySailent
https://github.com/flyboyhoward/MySailent.git
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 u2net_train_ddp.py
python -m torch.distributed.launch --nproc_per_node=2 u2net_train_ddp.py