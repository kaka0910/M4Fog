# ----------------------deeplab----------------------------
# CUDA_VISIBLE_DEVICES=0 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model deeplabv3p --checkname deeplabv3p \
#            --lr 0.0001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52120 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/01deeplabv3_norm/ \

# ----------------------scselinknet----------------------------
# CUDA_VISIBLE_DEVICES=1 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model scselink --checkname scselink \
#            --lr 0.0001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52010 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/02scselink_norm/ \


# ----------------------unet----------------------------
# CUDA_VISIBLE_DEVICES=0 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model unet --checkname unet \
#            --lr 0.0001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52030 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/03unet_norm/ \

# ----------------------unetpp----------------------------
# CUDA_VISIBLE_DEVICES=1 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model aspp_unet --checkname aspp_unet \
#            --lr 0.0001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52000 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/04unetpp_norm/ \

# ----------------------atten-unet----------------------------
# CUDA_VISIBLE_DEVICES=0 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model attu_net --checkname attu_net \
#            --lr 0.00001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:53010 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/05attenunet_norm/ \

# ----------------------abcnet----------------------------
# CUDA_VISIBLE_DEVICES=1 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model abcnet --checkname abcnet \
#            --lr 0.000001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:50020 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/06abcnet_norm/ \

# ----------------------banet----------------------------
CUDA_VISIBLE_DEVICES=0 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
           --model banet --checkname banet \
           --lr 0.00001 \
           --lr-scheduler 'poly' \
           --epochs 50 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52030 \
           --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/07banet_norm/ \

# ----------------------unetformer----------------------------
# CUDA_VISIBLE_DEVICES=0 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model unetformer --checkname unetformer \
#            --lr 0.00001 \
#            --lr-scheduler 'poly' \
#            --epochs 50 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52000 \
#            --model_savefolder /data2/sam-point-prompt-0202/output_17x_0224/Contrast_nc/08unetformer_norm/ \

# ----------------------dcswin----------------------------
# CUDA_VISIBLE_DEVICES=1 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model dcswin --checkname dcswin \
#            --lr 0.000005 \
#            --lr-scheduler 'poly' \
#            --epochs 100 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52010 \
#            --model_savefolder /data3/Seafog_HBH/cloud/output_final0524/09dcswin/ \

# ----------------------dlinkvit----------------------------
# CUDA_VISIBLE_DEVICES=3 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
#            --model dlinkvit_h8 --checkname dlinkvit_h8 \
#            --lr 0.00001 \
#            --lr-scheduler 'poly' \
#            --epochs 100 \
#            --weight-decay 0.05 \
#            --dist-url tcp://127.0.0.1:52030 \
#            --model_savefolder /data3/Seafog_HBH/cloud/output_final0524/10dlinkvit_h8/ \
