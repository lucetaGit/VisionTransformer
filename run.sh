#!/bin/bash


#  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 
  echo Pruning Threshold Traning!!

for pruning_pi in 0.00000001 
do
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.run \
      --rdzv-backend=c10d \
      --rdzv-endpoint=localhost:24900 \
      --nnodes=1 \
      --nproc-per-node=4 \
      main.py \
      --pretrained \
      --no-model-ema \
      --sched tanh \
      --num_workers 10 \
      --data-set IMNET \
      --model vit_base \
      --batch-size 16 \
      --epochs 2 \
      --output_dir /home/dongjin97/projects/deit_qat_share/ckpt/ \
      --quantize \
      --custom_pruning \
      --pruning_alpha 1 \
      --pruning_s 10 \
      --pruning_c 1000 \
      --pruning_k 100 \
      --pruning_pi ${pruning_pi}
done
      # --data-path /home/imagenet \

  # --world_size 2

  #--resume /home/isc/pytorch/ViT/deit_qat/ckpt/checkpoint.pth

  # --rdzv-backend=c10d             \
  # --rdzv-endpoint=localhost:24900 \
  # --nnodes=1 \
  # --nproc-per-node=4 \
  # --rdzv-backend=c10d             \
  # --rdzv-endpoint=localhost:24900 \

#ThreeAugment=False
#aa='rand-m9-mstd0.5-inc1'
#attn_only=False
#batch_size=128
#bce_loss=False
#clip_grad=None
#color_jitter=0.3
#cooldown_epochs=10
#cutmix=1.0
#cutmix_minmax=None
#data_path='/home/imagenet'
#data_set='IMNET'
#decay_epochs=30
#decay_rate=0.1
#device='cuda'
#dist_backend='nccl'
#dist_eval=False
#dist_url='env://'
#distillation_alpha=0.5
#distillation_tau=1.0
#distillation_type='none'
#distributed=True
#drop=0.0
#drop_path=0.1
#epochs=300
#eval=False
#eval_crop_ratio=0.875
#finetune=''
#gpu=0
#inat_category='name'

#input_size=224
#num_workers=10
#pin_mem=True

#lr=0.0005
#lr_noise=None
#lr_noise_pct=0.67
#lr_noise_std=1.0
#min_lr=1e-05
#mixup=0.8
#mixup_mode='batch'
#mixup_prob=1.0
#mixup_switch_prob=0.5
#model='vit_base_patch16_224'
#model_ema=True
#model_ema_decay=0.99996
#model_ema_force_cpu=False
#momentum=0.9
#opt='adamw'
#opt_betas=None
#opt_eps=1e-08
#output_dir='/home/isc/pytorch/ViT/deit/ckpt'
#patience_epochs=10
#pretrained=True
#rank=0
#recount=1
#remode='pixel'
#repeated_aug=True
#reprob=0.25
#resplit=False
#resume=''
#sched='cosine'
#seed=0
#smoothing=0.1
#src=False
#start_epoch=0
#teacher_model='regnety_160'
#teacher_path=''
#train_interpolation='bicubic'
#train_mode=True
#unscale_lr=False
#warmup_epochs=5
#warmup_lr=1e-06

#weight_decay=0.05
#world_size=2
#regularizer='kurt'
#reg_rate=1.0
#k=12.0
