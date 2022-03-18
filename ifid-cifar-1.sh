wandb offline

MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_32_07 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================1/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_32_07 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================2/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_32_07 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================3/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_39_29 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================4/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_39_29 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================5/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_39_29 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================6/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_40_00 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================7/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_40_00 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================8/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-DCGAN-train-2022_01_11_20_40_00 -cfg ./src/configs/CIFAR10/DCGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================9/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_02_57 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================10/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_02_57 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================11/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_02_57 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================12/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_16_06 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================13/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_16_06 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================14/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_16_06 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================15/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_19_13 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================16/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_19_13 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================17/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-GGAN-train-2022_01_11_21_19_13 -cfg ./src/configs/CIFAR10/GGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================18/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_11_21_20_08 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================19/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_11_21_20_08 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================20/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_11_21_20_08 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================21/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_24_23_36_10 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================22/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_24_23_36_10 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================23/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_24_23_36_10 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================24/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_11_21_22_26 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================25/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_11_21_22_26 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================26/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LGAN-train-2022_01_11_21_22_26 -cfg ./src/configs/CIFAR10/LGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================27/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_23_06 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================28/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_23_06 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================29/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_23_06 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================30/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_23_39 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================31/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_23_39 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================32/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_23_39 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================33/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_24_05 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================34/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_24_05 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================35/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-LSGAN-train-2022_01_11_21_24_05 -cfg ./src/configs/CIFAR10/LSGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================36/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_01_11_21_26_37 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================37/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_01_11_21_26_37 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================38/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_01_11_21_26_37 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================39/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_01_11_21_27_36 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================40/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_01_11_21_27_36 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================41/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_01_11_21_27_36 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================42/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_02_14_18_23_18 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================43/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_02_14_18_23_18 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================44/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-MHGAN-train-2022_02_14_18_23_18 -cfg ./src/configs/CIFAR10/MHGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================45/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_01_11_21_32_37 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================46/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_01_11_21_32_37 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================47/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_01_11_21_32_37 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================48/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_01_11_21_34_40 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================49/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_01_11_21_34_40 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================50/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_01_11_21_34_40 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================51/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_02_14_18_23_18 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================52/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_02_14_18_23_18 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================53/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-WC-train-2022_02_14_18_23_18 -cfg ./src/configs/CIFAR10/WGAN-WC.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================54/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_11_21_36_01 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================55/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_11_21_36_01 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================56/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_11_21_36_01 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================57/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_25_16_34_00 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================58/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_25_16_34_00 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================59/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_25_16_34_00 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================60/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_11_21_37_33 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================61/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_11_21_37_33 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================62/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-GP-train-2022_01_11_21_37_33 -cfg ./src/configs/CIFAR10/WGAN-GP.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================63/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_40_30 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================64/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_40_30 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================65/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_40_30 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================66/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_41_02 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================67/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_41_02 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================68/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_41_02 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================69/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_41_32 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================70/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_41_32 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================71/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-WGAN-DRA-train-2022_01_11_21_41_32 -cfg ./src/configs/CIFAR10/WGAN-DRA.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================72/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-ACGAN-Mod-train-2022_03_06_02_20_40 -cfg ./src/configs/CIFAR10/ACGAN-Mod.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================73/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-ACGAN-Mod-train-2022_03_06_02_20_40 -cfg ./src/configs/CIFAR10/ACGAN-Mod.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================74/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-ACGAN-Mod-train-2022_03_06_02_20_40 -cfg ./src/configs/CIFAR10/ACGAN-Mod.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================75/243 completed =============
MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python src/main.py --pre_resizer wo_resize --post_resizer tailored -std_stat -std_max 128 -std_step 128 -best --seed 1234 -ifid -data /root/ext_data/data/CIFAR10 -save ../studiogan/ -ckpt /root/ext_data/studiogan_ckpt/CIFAR10/checkpoints/CIFAR10-ACGAN-Mod-train-2022_03_06_02_22_36 -cfg ./src/configs/CIFAR10/ACGAN-Mod.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================76/243 completed =============
