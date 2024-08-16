# ----------------------unetpp----------------------------
CUDA_VISIBLE_DEVICES=0 python code/train_goes16.py --dataset goes16 --batch-size 2 --workers 1 \
           --model aspp_unet --checkname aspp_unet \
           --lr 0.0001 \
           --lr-scheduler 'poly' \
           --epochs 100 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52011 \
           --model_savefolder /nas2/data/users/xmq/cloud/GOES16/ \