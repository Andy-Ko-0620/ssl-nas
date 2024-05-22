python -m torch.distributed.launch --nproc_per_node=1 --master_port=29700 \
    main_pretrain.py --batch_size 64 \
    --world_size 1 \
    --accum_iter 16 \
    --model mae_vit_tiny_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1e-3 --weight_decay 0.05 \
    --output_dir logs/vit_tiny_in100_test_speed \
    --log_dir logs/vit_tiny_in100_test_speed \
    --data_path /app/Documents/ssl-nas/dataset/in100/ 