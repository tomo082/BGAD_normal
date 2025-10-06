import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
import albumentations as A
from .perlin import rand_perlin_2d_np
from .nsa import patch_ex
from .utils import excluding_images

VISA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]

class VisaDataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        # load dataset
        if excluded_images is not None:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
            self.x, self.y, self.mask, self.img_types = excluding_images(self.x, self.y, self.mask, self.img_types, excluded_images)
        else:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
        
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]
        
        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, y, mask, os.path.basename(img_path[:-4]), img_type

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, types = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.jpg')])
            x.extend(img_fpath_list)

            if img_type == 'normal':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                types.extend(['normal'] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask), list(types)


class VisaFSDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.reuse_times = 5
        
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.reuse_times
        self.a_labels = self.a_labels * self.reuse_times
        self.a_masks = self.a_masks * self.reuse_times
        
        self.transform_x = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]

        x = Image.open(img).convert('RGB')
        x = self.normalize(self.transform_x(x))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, label, mask

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))
        for type_ in ano_types:
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.jpg')])
            
            if type_ == 'normal':
                continue
            else:
                random.shuffle(img_fpath_list)
                num_anomalies_to_take = min(self.anomaly_nums, len(img_fpath_list))
                a_img_paths.extend(img_fpath_list[:num_anomalies_to_take])
                a_labels.extend([1] * num_anomalies_to_take)
                
                gt_type_dir = os.path.join(gt_dir, type_)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:num_anomalies_to_take]]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png') for img_fname in img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'normal')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)
                                 if f.endswith('.png') or f.endswith('.jpg')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisaFSCopyPasteDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.repeat_num = 10
        self.reuse_times = 5

        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        
        self.a_imgs = self.a_imgs * self.repeat_num
        self.a_labels = self.a_labels * self.repeat_num
        self.a_masks = self.a_masks * self.repeat_num
        
        self.transform_img = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
            
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
        
        self.augmentors = [A.RandomRotate90(), A.Flip(), A.Transpose(),
                           A.OneOf([A.GaussNoise(), A.GaussNoise()], p=0.2),
                           A.OneOf([A.MotionBlur(p=.2), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1)], p=0.2),
                           A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                           A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=.1), A.PiecewiseAffine(p=0.3)], p=0.2),
                           A.OneOf([A.CLAHE(clip_limit=2), A.Sharpen(), A.Emboss(), A.RandomBrightnessContrast()], p=0.3),
                           A.HueSaturationValue(p=0.3)]
                           
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
    
    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
            
            # Apply copy-paste only to a subset of anomaly images to generate more diverse samples
            if idx >= len(self.n_imgs) + len(self.a_imgs) // self.repeat_num * self.reuse_times:
                img, mask = self.copy_paste(img, mask)
                img, mask = Image.fromarray(img), Image.fromarray(mask)
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
            img = Image.open(img).convert('RGB')

        img = self.normalize(self.transform_img(img))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            if isinstance(mask, str):
                mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return img, label, mask

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors)), 3, replace=False)
        aug = A.Compose([self.augmentors[aug_ind[0]], self.augmentors[aug_ind[1]], self.augmentors[aug_ind[2]]])
        return aug

    def copy_paste(self, img_path, mask_path):
        n_idx = np.random.randint(len(self.n_imgs))
        aug = self.randAugmenter()

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_image = cv2.imread(self.n_imgs[n_idx])
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
        
        mask = np.asarray(Image.open(mask_path))
        
        augmented = aug(image=image, mask=mask)
        aug_image, aug_mask = augmented['image'], augmented['mask']
        
        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
        return n_image, aug_mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []

        img_dir_test = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        
        ano_types = [d for d in sorted(os.listdir(img_dir_test)) if os.path.isdir(os.path.join(img_dir_test, d)) and d != 'normal']
        
        for type_ in ano_types:
            img_type_dir = os.path.join(img_dir_test, type_)
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith(('.png', '.jpg'))])
            
            random.shuffle(img_fpath_list)
            num_anomalies_to_take = min(self.anomaly_nums, len(img_fpath_list))
            selected_imgs = img_fpath_list[:num_anomalies_to_take]
            a_img_paths.extend(selected_imgs)
            a_labels.extend([1] * len(selected_imgs))
            
            gt_type_dir = os.path.join(gt_dir, type_)
            img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in selected_imgs]
            gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png') for img_fname in img_fname_list]
            a_mask_paths.extend(gt_fpath_list)

        img_dir_train = os.path.join(self.dataset_path, self.class_name, 'train', 'normal')
        img_fpath_list_train = sorted([os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train) if f.endswith(('.png', '.jpg'))])
        n_img_paths.extend(img_fpath_list_train)
        n_labels.extend([0] * len(img_fpath_list_train))
        n_mask_paths.extend([None] * len(img_fpath_list_train))
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisaPseudoDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.repeat_num = 10
        
        self.n_imgs, _, _, _, _, _ = self.load_dataset_folder()
        self.a_imgs = self.n_imgs * self.repeat_num
        
        self.transform_img_np = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size)])
        
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(c.norm_mean, c.norm_std)])
        self.anomaly_source_paths = sorted(glob.glob(c.anomaly_source_path + "/*/*.jpg"))
        
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                           iaa.Solarize(0.5, threshold=(32,128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.cropsize[1], self.cropsize[0]))
        anomaly_img_augmented = aug(image=anomaly_source_img)
        
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.cropsize[0], self.cropsize[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        beta = torch.rand(1).numpy()[0] * 0.8
        
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        
        has_anomaly = 1 if np.sum(msk) > 0 else 0
        return augmented_image, msk, has_anomaly

    def transform_image(self, image_path, anomaly_source_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform_img_np(image)
        image = np.asarray(image)
        
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = augmented_image.astype(np.uint8)
        
        return self.normalize(augmented_image), torch.from_numpy(np.transpose(anomaly_mask, (2, 0, 1))), has_anomaly

    def __getitem__(self, idx):
        if idx < len(self.n_imgs):
            img_path, label, mask = self.n_imgs[idx], 0, torch.zeros([1, self.cropsize[0], self.cropsize[1]])
            image = self.normalize(self.transform_img_np(Image.open(img_path).convert('RGB')))
            return image, label, mask
        else:
            n_idx = np.random.randint(len(self.n_imgs))
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            image, mask, label = self.transform_image(self.n_imgs[n_idx], self.anomaly_source_paths[anomaly_source_idx])
            return image, label, mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []
        
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'normal')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)
                                 if f.endswith('.png') or f.endswith('.jpg')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisaAnomalyDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES
        self.copy_paste_dataset = VisaFSCopyPasteDataset(c, is_train=is_train)
        self.pseudo_dataset = VisaPseudoDataset(c, is_train=is_train)

    def __len__(self):
        return len(self.copy_paste_dataset) + len(self.pseudo_dataset) - len(self.copy_paste_dataset.n_imgs)

    def __getitem__(self, idx):
        if idx >= len(self.pseudo_dataset):
            return self.copy_paste_dataset.__getitem__(idx)
        else:
            return self.pseudo_dataset.__getitem__(idx)
