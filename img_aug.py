import imgaug.augmenters as iaa
import os
from PIL import Image
import argparse

# Define argument parser
parser = argparse.ArgumentParser(description='Apply image augmentation to images in a folder.')
parser.add_argument('--path', type=str, help='path to images folder', required=True)

# Define augmentation sequence
aug = iaa.Sequential([
    iaa.Fliplr(p=0.5), # horizontal flips
    iaa.Affine(rotate=(-10, 10)), # rotations
    iaa.Affine(scale=(0.5, 1.5)), # scaling
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}), # translation
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # noise addition
])

# Parse arguments
args = parser.parse_args()
folder_path = args.path

# Loop through images in folder and apply augmentation
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # Apply augmentation
        img_aug = aug(image=img)

        # Save augmented image
        img_aug_path = os.path.join(folder_path, "aug_" + filename)
        img_aug.save(img_aug_path)
