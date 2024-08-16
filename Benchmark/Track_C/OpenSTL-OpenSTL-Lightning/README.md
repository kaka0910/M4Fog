## Track C Benchmark

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
