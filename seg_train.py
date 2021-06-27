import os
import numpy as np
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imread

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import albumentations as albu
import re

import torch
import numpy as np
import segmentation_models_pytorch as smp

# Ensure usage of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Dataset(BaseDataset):
    """Hair segmentation. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
        dataset_type (False, True): switch between figaro dataset and celebA dataset
            it needs a different type of indexing.

    """

    CLASSES = ['hair', 'no_hair']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        try:
            self.mask_ids = sorted(os.listdir(masks_dir))

            # Sort file names on index only to get matchin origninal and mask image
            images_list = sorted(os.listdir(images_dir))
            self.im_ids = sorted(images_list, key = lambda x: int(x.strip('.jpg')))

            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.im_ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

            # convert str names to class values on masks
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

            self.augmentation = augmentation
            self.preprocessing = preprocessing
        except:
            self.im_ids = sorted(os.listdir(images_dir))
            self.mask_ids = sorted(os.listdir(masks_dir))

            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.im_ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

            # convert str names to class values on masks
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

            self.augmentation = augmentation
            self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # This did not work
        mask = cv2.imread(self.masks_fps[i], 0)

        # Reshaping if mask image is not the same size as original image
        m_mask = mask.shape
        image = cv2.resize(image, (m_mask[1], m_mask[0]))

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # Invert image for preprocessing, so augmentation will be correct
        mask = 1-mask

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.im_ids)

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.RandomCrop(height=512, width=512, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 512, always_apply=True, border_mode=0),
        albu.Resize(512, 512, interpolation=1)

    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.RandomCrop(height=512, width=512, always_apply=True),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def initialize_model(model_name):

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['hair']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    if model_name == 'Unet':
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif model_name == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif model_name == 'PAN':
        model = smp.PAN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif model_name == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    return model, loss, metrics, optimizer, preprocessing_fn


def init_datasets(x_train_dir, y_train_dir, x_val_dir, y_val_dir, preprocessing_fn):
    CLASSES = ['hair']
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_val_dir,
        y_val_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    return train_dataset, valid_dataset

def init_DataLoader(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, valid_loader

# Create possibility to change easily on what to run on GPU and CPU
# e.g. run train on GPU and validate on CPU
def epochs(model, loss, metrics, optimizer, device):
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    return train_epoch, valid_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40,
        help="Number of epochs to train the model")
    parser.add_argument("--plot", help="Store the loss and IOU score plots",
        action="store_true")
    parser.add_argument("--model_name", type=str, default="best_model",
        help="Name of the best model resulting from training")
    parser.add_argument("--data_type", type=str, default="figaro",
        help="Choose between figaro dataset and celebA dataset, --data celeb / --data figaro")
    parser.add_argument("--model", type=str, default='Unet',
        help="Choose architecture, Unet, FPN, PAN, DeepLabV3, UnetPlusPlus")
    parser.add_argument("--batch_size", type=int, default=4,
        help="Specify training batch_size this is depenend on memory capabilitys of GPU")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
        print("# Using {} GPU backend".format(torch.cuda.get_device_name(device)))
    else:
        device= 'cpu'
        print("# Using CPU backend")

    model_name = args.model_name + ".pth"

    if not os.path.exists('models/'):
        os.makedirs('models/')
    if not os.path.exists('results/'):
        os.makedirs('results/')

    if args.data_type == 'figaro':
        DATA_DIR = 'Data//Figaro1k/'
        x_train_dir = os.path.join(DATA_DIR, 'Original/Training')
        y_train_dir = os.path.join(DATA_DIR, 'GT/Training')

        x_val_dir = os.path.join(DATA_DIR, 'Original/val')
        y_val_dir = os.path.join(DATA_DIR, 'GT/valannot')

        x_test_dir = os.path.join(DATA_DIR, 'Original/Testing')
        y_test_dir = os.path.join(DATA_DIR, 'GT/Testing')
        # x_train_dir = 'Data/Figaro1k/Original/Training'
        # y_train_dir = 'Data/Figaro1k/GT/Training'
        #
        # x_test_dir = 'Data/Figaro1k/Original/Testing'
        # y_test_dir = 'Data/Figaro1k/GT/Testing'
        #
        # x_val_dir = 'Data/Figaro1k/Original/val'
        # y_val_dir ='Data/Figaro1k/GT/valannot'

    elif args.data_type == 'celeb':
        x_train_dir = 'Data/Celeb1k/train/original'
        y_train_dir = 'Data/Celeb1k/train/mask'

        x_val_dir = 'Data/Celeb1k/val/original'
        y_val_dir = 'Data/Celeb1k/val/mask'

        x_test_dir = 'Data/Celeb1k/test/original'
        y_test_dir ='Data/Celeb1k/test/original'


    print("# Initializing model...")
    model, loss, metrics, optimizer, preprocessing_fn = initialize_model(args.model)

    print("# Initializing datasets...")
    train_dataset, valid_dataset = init_datasets(
                                        x_train_dir,
                                        y_train_dir,
                                        x_val_dir,
                                        y_val_dir,
                                        preprocessing_fn)

    print("# Initializing data loaders...")
    train_loader, valid_loader = init_DataLoader(train_dataset, valid_dataset, args.batch_size)

    print("# Initializing epoch runners...")
    train_epoch, valid_epoch = epochs(model, loss, metrics, optimizer, device)

    print("# Starting training...")

    train_loss = []
    train_iou = []
    valid_loss = []
    valid_iou = []

    max_score = 0

    for i in range(0, args.epochs):
        print('\nEpoch: {}'.format(i))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_iou.append(train_logs['iou_score'])
        train_loss.append(train_logs['dice_loss'])

        valid_iou.append(valid_logs['iou_score'])
        valid_loss.append(valid_logs['dice_loss'])

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, 'models/best_model_celeb_unet.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    if args.plot:
        plt.plot(train_iou, label='training iou')
        plt.plot(valid_iou, label='validation iou')
        plt.title('IOU')
        plt.xlabel('epoch')
        plt.ylabel('iou score')
        plt.legend()
        plt.savefig('results/IOU_score' + args.model_name +  '.png')

        plt.plot(train_loss, label='training loss')
        plt.plot(valid_loss, label='validation loss')
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('iou score')
        plt.legend()
        plt.savefig('results/loss' + args.model_name +  '.png')
