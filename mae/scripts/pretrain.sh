python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 \
    main_pretrain.py --batch_size 64 \
    --world_size 1 \
    --accum_iter 16 \
    --model mae_vit_tiny_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1e-3 --weight_decay 0.05 \
    --data_path /home/ubuntu/Documents/dataset/imagenet1k/ 