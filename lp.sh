
#BigGAN-Mod
python src/main.py -lp -c src/configs/CIFAR10/unconditional-BigGAN-Mod.json --checkpoint_folder checkpoints/unconditional-BigGAN-Mod-train-2021_08_18_11_17_04/

#BigGAN-CR
python src/main.py -lp -c src/configs/CIFAR10/unconditional-BigGAN-Mod-CR.json --checkpoint_folder checkpoints/unconditional-BigGAN-Mod-CR-train-2021_08_18_11_17_34/

#BigGAN-Mod-CR-SimCLR
python src/main.py -lp -c src/configs/CIFAR10/unconditional-BigGAN-Mod-CR-SimCLR.json --checkpoint_folder checkpoints/unconditional-BigGAN-Mod-CR-SimCLR-train-2021_08_23_18_26_34/

#BigGAN-Mod-only_cr_aug
python src/main.py -lp -c src/configs/CIFAR10/unconditional-BigGAN-Mod-use_only_cr_aug.json --checkpoint_folder checkpoints/unconditional-BigGAN-Mod-use_only_cr_aug-train-2021_08_28_06_56_21/

#BigGAN-Mod-BYOL-no_bn
python src/main.py -lp -c src/configs/CIFAR10/unconditional-BigGAN-Mod-BYOL.json --checkpoint_folder checkpoints/unconditional-BigGAN-Mod-BYOL-train-2021_08_24_14_56_42/

#BigGAN-Mod-BYOL-bn
python src/main.py -lp -c src/configs/CIFAR10/unconditional-BigGAN-Mod-BYOL.json --checkpoint_folder checkpoints/unconditional-BigGAN-Mod-BYOL-train-2021_08_24_14_58_07/

