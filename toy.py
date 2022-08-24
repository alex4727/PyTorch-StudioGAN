# Imports
from PIL import Image
from argparse import ArgumentParser
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from num2words import num2words

def get_random_locations(upper_left_corner, lower_right_corner, img_size):
    # corners in (x, y) format 
    # note that in cv2, dimensions are hwc.
    x = random.randint(upper_left_corner[0], lower_right_corner[0] - img_size + 1)
    y = random.randint(upper_left_corner[1], lower_right_corner[1] - img_size + 1)
    return x, y

def randomize_image_location(n, images, image_size):
    bg = cv2.imread("./black.png")
    fg_images = []

    if image_size == "random":
        for image in images:
            img_size = random.randint(128, 512)
            img = Image.open(image).resize((img_size, img_size), Image.Resampling.LANCZOS)
            fg_images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    else:
        image_size = int(image_size)
        for image in images:
            img = Image.open(image).resize((image_size, image_size), Image.Resampling.LANCZOS)
            fg_images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    
    corners = {"1": [(0, 0, 1023, 1023)], \
        "2v": [(0, 0, 511, 1023), (512, 0, 1023, 1023)], \
        "2h": [(0, 0, 1023, 511), (0, 512, 1023, 1023)], \
        "4": [(0, 0, 511, 511), (512, 0, 1023, 511), (0, 512, 511, 1023), (512, 512, 1023, 1023)]}

    boxes = corners[str(n)]

    for i, box in enumerate(boxes):
        x, y = get_random_locations((box[0], box[1]), (box[2], box[3]), fg_images[i].shape[0])
        bg[y:y+fg_images[i].shape[0], x:x+fg_images[i].shape[1]] = fg_images[i]
    return bg

def preprare_arguments():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("-n", "--num_images", type=str, help="number of images in an image", default=1)
    parser.add_argument("-s", "--image_size", type=str, help="image size", default="random")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = preprare_arguments()
    n = args.num_images
    img_size = args.image_size

    setA = "/root/data/data/sample_fd/FFHQ_SAMPLE/setA"
    setB = "/root/data/data/sample_fd/FFHQ_SAMPLE/setB"
    base = "/root/data/data/sample_fd/"

    if n == "2h":
        save_path = os.path.join(base, f"twoh_{img_size}")
    elif n == "2v":
        save_path = os.path.join(base, f"twov_{img_size}")
    else:
        save_path = os.path.join(base, f"{num2words(int(n))}_{img_size}")

    setA_save_path = save_path+"/setA/all"
    setB_save_path = save_path+"/setB/all"
    os.makedirs(setA_save_path)
    os.makedirs(setB_save_path)

    images_A = [os.path.join(setA, img) for img in sorted(os.listdir(setA))]
    images_B = [os.path.join(setB, img) for img in sorted(os.listdir(setB))]
    A_queue = images_A.copy()
    B_queue = images_B.copy()
    random.shuffle(A_queue)
    random.shuffle(B_queue)

    print("Creating Images for setA")
    for i in tqdm(range(0, 35000)):
        tmp = []
        for j in range(0, 2 if (n == "2v" or n == "2h") else int(n)):
            try:
                tmp.append(A_queue.pop())
            except:
                print("Popped all images from setA")
                A_queue = images_A.copy()
                random.shuffle(A_queue)
                tmp.append(A_queue.pop())
        new_img = randomize_image_location(n, tmp, img_size)
        cv2.imwrite(os.path.join(setA_save_path, "%05d.png" % i), new_img)
    
    print("Creating Images for setB")
    for i in tqdm(range(0, 35000)):
        tmp = []
        for j in range(0, 2 if (n == "2v" or n == "2h") else int(n)):
            try:
                tmp.append(B_queue.pop())
            except:
                print("Popped all images from setB")
                B_queue = images_B.copy()
                random.shuffle(B_queue)
                tmp.append(B_queue.pop())
        new_img = randomize_image_location(n, tmp, img_size)
        cv2.imwrite(os.path.join(setB_save_path, "%05d.png" % i), new_img)
