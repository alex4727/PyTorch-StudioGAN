wandb offline
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SNGAN-train-2022_03_09_03_16_06 -cfg ./src/configs/Baby_ImageNet/SNGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================1/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SNGAN-train-2022_03_09_03_16_06 -cfg ./src/configs/Baby_ImageNet/SNGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================2/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SNGAN-train-2022_03_09_03_16_06 -cfg ./src/configs/Baby_ImageNet/SNGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================3/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SAGAN-train-2022_03_09_17_06_13 -cfg ./src/configs/Baby_ImageNet/SAGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================4/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SAGAN-train-2022_03_09_17_06_13 -cfg ./src/configs/Baby_ImageNet/SAGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================5/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SAGAN-train-2022_03_09_17_06_13 -cfg ./src/configs/Baby_ImageNet/SAGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================6/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SAGAN-train-2022_03_09_17_32_59 -cfg ./src/configs/Baby_ImageNet/SAGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================7/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SAGAN-train-2022_03_09_17_32_59 -cfg ./src/configs/Baby_ImageNet/SAGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================8/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 256 -std_step 256 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-SAGAN-train-2022_03_09_17_32_59 -cfg ./src/configs/Baby_ImageNet/SAGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================9/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-BigGAN-train-2022_02_12_23_06_42 -cfg ./src/configs/Baby_ImageNet/BigGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================10/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-BigGAN-train-2022_02_12_23_06_42 -cfg ./src/configs/Baby_ImageNet/BigGAN.yaml --eval_backbone SwAV_torch  -metrics none
echo ==================11/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-BigGAN-train-2022_02_12_23_06_42 -cfg ./src/configs/Baby_ImageNet/BigGAN.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================12/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored  -std_stat -std_max 1024 -std_step 1024 -best --seed 1234 -ifid -data /home/mgkang/data/Baby_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Baby_ImageNet/checkpoints/Baby_ImageNet-BigGAN-train-2022_02_19_21_22_58 -cfg ./src/configs/Baby_ImageNet/BigGAN.yaml --eval_backbone InceptionV3_tf  -metrics none
echo ==================13/39 completed =============
python src/main.py --pre_resizer lanczos --post_resizer tailored -best --seed 1234 -ifid -mpc  -data /home/mgkang/data/Grandpa_ImageNet -save ../studiogan/ -ckpt /home/mgkang/joong/eval_temp/studiogan_ckpts/Grandpa_ImageNet/checkpoints/Grandpa_ImageNet-StyleGAN2-SPD-train-2022_02_24_13_43_37 -cfg ./src/configs/Grandpa_ImageNet/StyleGAN2-SPD.yaml --eval_backbone Swin-T_torch  -metrics none
echo ==================30/36 completed =============
