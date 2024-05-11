# python -m torch.distributed.launch --nproc_per_node=1 --master_port=25678 main_dino.py \
#        --data_path /app/Documents/NASwithSSL/sub-imagenet --output_dir logs/ \
#        --arch deit_tiny --batch_size_per_gpu 256  --momentum_teacher 0.9995 \
#        --global_crops_scale 0.14 1 --local_crops_number 0 &> dino-log/training-log.txt &

python -m torch.distributed.launch --nproc_per_node=1 main_dino.py \
       --data_path /app/Documents/NASwithSSL/sub-imagenet --output_dir logs/local-crop-2-bs-128-vit_small \
       --arch vit_small --batch_size_per_gpu 128  --momentum_teacher 0.9995 \
       --local_crops_number 2 &> logs/local-crop-2-bs-128-vit_small/training-log.txt &