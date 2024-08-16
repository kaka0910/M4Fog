# ----------------------unet----------------------------
# CUDA_VISIBLE_DEVICES=1 python code/train_fy4a.py --dataset fy4a --batch-size 2 --workers 1 \
#            --model unet --checkname unet \
#            --lr 0.0001 \
#            --lr-scheduler 'poly' \
#            --epochs 100 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52011 \
#            --model_savefolder /nas2/data/users/xmq/cloud/FY4A/ \

# ----------------------unetpp----------------------------
CUDA_VISIBLE_DEVICES=0 python code/train_fy4a.py --dataset fy4a --batch-size 2 --workers 1 \
           --model aspp_unet --checkname aspp_unet \
           --lr 0.0001 \
           --lr-scheduler 'poly' \
           --epochs 100 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52031 \
           --model_savefolder /nas2/data/users/xmq/cloud/FY4A/unetpp/ \