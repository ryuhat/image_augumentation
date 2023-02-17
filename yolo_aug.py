import imgaug.augmenters as iaa
import os
from PIL import Image
import argparse

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to the image folder")
parser.add_argument("--label", required=True, help="Path to the label folder")
args = parser.parse_args()

# Define augmentation sequence
aug = iaa.Sequential([
    iaa.Fliplr(p=0.5), # horizontal flips
    iaa.Affine(rotate=(-10, 10)), # rotations
    iaa.Affine(scale=(0.5, 1.5)), # scaling
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}), # translation
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # noise addition
])

# Define path to images and labels
img_folder = args.image
label_folder = args.label

# Loop through images in folder and apply augmentation
for filename in os.listdir(img_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img_path = os.path.join(img_folder, filename)
        img = Image.open(img_path)

        # Load YOLO label file
        label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Extract bounding box coordinates
        bboxes = []
        for line in lines:
            bbox = [float(x) for x in line.split()[1:]]
            bboxes.append(bbox)

        # Create BoundingBoxesOnImage object
        bbsoi = iaa.BoundingBoxesOnImage.from_xyxy_array(bboxes, shape=img.size)

        # Apply augmentation
        img_aug, bbsoi_aug = aug(image=img, bounding_boxes=bbsoi)

        # Update bounding box coordinates
        bboxes_aug = bbsoi_aug.to_xyxy_array()

        # Save augmented image
        img_aug_path = os.path.join(img_folder, "aug_" + filename)
        img_aug.save(img_aug_path)

        # Save augmented YOLO label file
        label_aug_path = os.path.join(label_folder, "aug_" + os.path.splitext(filename)[0] + ".txt")
        with open(label_aug_path, "w") as f:
            for i in range(len(bboxes_aug)):
                x1, y1, x2, y2 = bboxes_aug[i]
                label = lines[i].split()[0]
                f.write(f"{label} {x1} {y1} {x2} {y2}\n")
