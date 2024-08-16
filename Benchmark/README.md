## M4Fog benchmarks

### Track-A benchmark

Static marine fog detection refers to identifying and delineating marine fog areas at a specific moment based on super data cubes from that moment. It can support the input of either multi-channel geostationary satellite data alone or in combination with sea surface temperature and constant data. For more details on Track A, please refer to the Track A README. Here is an example of single GPU non-distributed training UNetpp on Himawari-8/9 dataset.

```shell
# Training on H8/9
CUDA_VISIBLE_DEVICES=1 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
           --model aspp_unet --checkname aspp_unet \
           --lr 0.0001 \
           --epochs 100 \
           --weight-decay 0.05 \
           --model_savefolder /nas/cloud/H8/ \
```

### Track-B benchmark

Track B dynamic marine fog detection refers to identifying and delineating marine fog areas at a specific moment by utilizing continuous super data cubes. We also provide the code for calculate the motion feature between adjacent time data. For more details on Track B, please refer to the Track B README. Here is an example of single GPU non-distributed training UNetpp on Himawari-8/9 dataset with motion data.

```shell
# Training on H8/9 with motion data
CUDA_VISIBLE_DEVICES=1 python code/train_h8_motion.py \
            --dataset h8_motion_v2 --batch-size 16 --workers 1 \
            --model unet_motion_v2 --checkname unet_motion_v2 \
            --model_optimizer SGD \
            --lr 0.0001 \
            --epochs 100 \
            --weight-decay 0.05 \
            --model_savefolder /motion/01unet_concatdist_v2/ \
```

### Track-C benchmark

For the experiments, the OpenSTL framework is selected as the foundation benchmark. Based on this codebase, we supply the process M4Fog dataset and generate corresponding data. We also provide MSE, MAE, SSIM and PSNR as evaluation metrics to measure model performance. For more details on Track C, please refer to the Track C README. Here is an example of single GPU non-distributed training PredRNN on Meteo-single-area dataset.

```shell
cd OpenSTL-OpenSTL-Lightning

# Training on meteo
CUDA_VISIBLE_DEVICES=0 python tools/train.py -d meteo -c configs/meteo/PredRNN.py --ex_name meteo_supply_predrnn
# Test on meteo
CUDA_VISIBLE_DEVICES=0 python tools/test.py -d meteo -c configs/meteo/PredRNN.py --ex_name meteo_supply_predrnn --test
# Finetune on meteo
CUDA_VISIBLE_DEVICES=1 python tools/train.py -d meteo -c configs/meteo/PredRNN.py --ex_name meteo_supply_finetue --ckpt_path meteo_supply_predrnn
```
