import albumentations as A
import cv2
import os
from tqdm import tqdm
import numpy as np

def augmentation_pipeline():
    image_size = 224  

    transforms_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(image_size, image_size),
        A.CoarseDropout(max_holes=1, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), fill_value=0, p=0.7),
        
    ])

    return transforms_train

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def augment_and_save_images(input_dir,transforms_train):
    
    # List all images in the input directory
    image_filenames = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', 'JPG'))]
    
    for img_name in tqdm(image_filenames):
        # Load the image
        image_path = os.path.join(input_dir, img_name)
        image = load_image(image_path)

        augmented = transforms_train(image=image)
        augmented_image = augmented['image']

        # Convert augmented image back to BGR for saving with OpenCV
        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        # Save augmented image in the output directory using the same name
        # output_path = os.path.join(output_dir, img_name)
        # cv2.imwrite(output_path, augmented_image_bgr)


# input_dir = r''  
# output_dir = r''  #

# augment_and_save_images(input_dir, output_dir, transforms_train)
