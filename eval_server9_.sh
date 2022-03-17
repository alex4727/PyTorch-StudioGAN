wandb offline
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_03_19_50 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================1/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_03_19_50 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================2/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_03_19_50 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================3/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_03_19_50 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================4/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_03_19_50 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================5/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_22_13_02 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================6/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_22_13_02 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================7/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_22_13_02 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================8/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_22_13_02 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================9/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SNGAN-train-2022_03_08_22_13_02 -cfg ./src/configs/Papa_ImageNet/SNGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================10/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SAGAN-train-2022_03_09_00_55_23 -cfg ./src/configs/Papa_ImageNet/SAGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================11/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SAGAN-train-2022_03_09_00_55_23 -cfg ./src/configs/Papa_ImageNet/SAGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================12/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SAGAN-train-2022_03_09_00_55_23 -cfg ./src/configs/Papa_ImageNet/SAGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================13/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SAGAN-train-2022_03_09_00_55_23 -cfg ./src/configs/Papa_ImageNet/SAGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================14/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 256 -std_step 256 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-SAGAN-train-2022_03_09_00_55_23 -cfg ./src/configs/Papa_ImageNet/SAGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================15/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_07 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================16/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_07 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================17/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_07 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================18/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_07 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================19/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_07 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================20/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_09 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================21/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_09 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================22/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_09 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================23/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_09 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================24/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-BigGAN-train-2022_02_23_16_26_09 -cfg ./src/configs/Papa_ImageNet/BigGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================25/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_18_16_43_49 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================26/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_18_16_43_49 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================27/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_18_16_43_49 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================28/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_18_16_43_49 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================29/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_18_16_43_49 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================30/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_28_07_20_51 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================31/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_28_07_20_51 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================32/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_28_07_20_51 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================33/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_28_07_20_51 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================34/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ContraGAN-train-2022_02_28_07_20_51 -cfg ./src/configs/Papa_ImageNet/ContraGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================35/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_18_20_16_51 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================36/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_18_20_16_51 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================37/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_18_20_16_51 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================38/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_18_20_16_51 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================39/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_18_20_16_51 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================40/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_21_15_09_16 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================41/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_21_15_09_16 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================42/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_21_15_09_16 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================43/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_21_15_09_16 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================44/60 completed =============
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py --pre_resizer lanczos --post_resizer tailored -sync_bn -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-ReACGAN-train-2022_02_21_15_09_16 -cfg ./src/configs/Papa_ImageNet/ReACGAN.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================45/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_18_19_49_49 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================46/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_18_19_49_49 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================47/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_18_19_49_49 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================48/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_18_19_49_49 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================49/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_18_19_49_49 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================50/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_20_14_40_08 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================51/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_20_14_40_08 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================52/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_20_14_40_08 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================53/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_20_14_40_08 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================54/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN2-SPD-train-2022_02_20_14_40_08 -cfg ./src/configs/Papa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================55/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN3-t-SPD-train-2022_02_24_19_24_04 -cfg ./src/configs/Papa_ImageNet/StyleGAN3-t-SPD.yaml --eval_backbone InceptionV3_tf  -metrics is fid  -ref valid 
echo ==================56/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN3-t-SPD-train-2022_02_24_19_24_04 -cfg ./src/configs/Papa_ImageNet/StyleGAN3-t-SPD.yaml --eval_backbone SwAV_torch  -metrics is fid prdc  -ref train 
echo ==================57/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN3-t-SPD-train-2022_02_24_19_24_04 -cfg ./src/configs/Papa_ImageNet/StyleGAN3-t-SPD.yaml --eval_backbone SwAV_torch  -metrics is fid  -ref valid 
echo ==================58/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN3-t-SPD-train-2022_02_24_19_24_04 -cfg ./src/configs/Papa_ImageNet/StyleGAN3-t-SPD.yaml --eval_backbone Swin-T_torch  -metrics is fid prdc  -ref train 
echo ==================59/60 completed =============
CUDA_VISIBLE_DEVICES=0,1 python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -mpc -DDP -data /root/ext_data/data/Papa_ImageNet -save ../studiogan/ --num_eval 1 -ckpt /root/ext_data/studiogan_ckpt/Papa_ImageNet/checkpoints/Papa_ImageNet-StyleGAN3-t-SPD-train-2022_02_24_19_24_04 -cfg ./src/configs/Papa_ImageNet/StyleGAN3-t-SPD.yaml --eval_backbone Swin-T_torch  -metrics is fid  -ref valid 
echo ==================60/60 completed =============