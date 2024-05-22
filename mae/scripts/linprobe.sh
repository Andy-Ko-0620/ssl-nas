python -m torch.distributed.launch --nproc_per_node=1 --master_port=29900 \
    main_finetune.py --batch_size 128 \
    --world_size 1 \
    --accum_iter 8 \
    --epochs 100 \
    --model vit_tiny_patch16 \
    --drop_path 0 \
    --warmup_epochs 40 \
    --mixup 0.2 \
    --blr 1e-3 --weight_decay 0.05 \
    --data_path /app/Documents/ssl-nas/dataset/in100/ \
    --nb_classes 100 \
    --finetune output_dir/best.pth \
    --output_dir logs/vit_tiny_finetune_ckpt-pretrain_e300 \
    --log_dir logs/vit_tiny_finetune_ckpt-pretrain_e300  