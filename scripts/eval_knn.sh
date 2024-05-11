# python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
#        --pretrained_weights /app/Documents/NASwithSSL/logs/local-crop-0-bs-256/checkpoint.pth --checkpoint_key teacher \
#        --arch vit_tiny --data_path /app/Documents/NASwithSSL/sub-imagenet/ --num_classes 100

python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py \
       --arch vit_small --data_path /media/vms/imagenet --num_classes 1000