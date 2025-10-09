import os
import numpy as np
import torch
import warnings
from config import parse_args
from datasets import MVTEC_CLASS_NAMES, BTAD_CLASS_NAMES, VISA_CLASS_NAMES
from utils.utils import init_seeds


def main_single(args):
    # model path
    args.model_path = "{}_{}_{}_{}".format(
        args.dataset, args.backbone_arch, args.flow_arch, args.class_name)
    
    # image
    args.img_size = (args.inp_size, args.inp_size)  
    args.crop_size = (args.inp_size, args.inp_size)  
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    args.img_dims = [3] + list(args.img_size)

    # output settings
    args.save_results = True
    
    from engines.bgad_test_engine import test
    img_auc, pix_auc, pro_auc = test(args)

    return img_auc, pix_auc, pro_auc


import os
import numpy as np
import torch
import warnings
from config import parse_args
from datasets import MVTEC_CLASS_NAMES, BTAD_CLASS_NAMES, VISA_CLASS_NAMES
from utils.utils import init_seeds


def main_single(args):
    # ... (main_single 関数の内容は変更なし) ...
    args.model_path = "{}_{}_{}_{}".format(
        args.dataset, args.backbone_arch, args.flow_arch, args.class_name)
    
    # image
    args.img_size = (args.inp_size, args.inp_size)  
    args.crop_size = (args.inp_size, args.inp_size)  
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    args.img_dims = [3] + list(args.img_size)

    # output settings
    args.save_results = True
    
    from engines.bgad_test_engine import test
    img_auc, pix_auc, pro_auc = test(args)

    return img_auc, pix_auc, pro_auc


def main():
    init_seeds()
    # args はこの関数の中で定義されています
    args = parse_args()

    # setting cuda 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda")

    img_aucs, pixel_aucs, pro_aucs = [], [], []

    # ▼▼▼▼▼ この部分が main 関数の中にある必要があります ▼▼▼▼▼
    if args.class_name == 'none':  # default training all classes
        if args.dataset == 'mvtec':
            CLASS_NAMES = MVTEC_CLASS_NAMES
        elif args.dataset == 'btad':
            CLASS_NAMES = BTAD_CLASS_NAMES
        elif args.dataset == 'visa':
            CLASS_NAMES = VISA_CLASS_NAMES
        else:
            CLASS_NAMES = [] # エラーを防ぐため
    else:
        CLASS_NAMES = [args.class_name]
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    for class_name in CLASS_NAMES:
        args.class_name = class_name
        img_auc, pix_auc, pro_auc = main_single(args)
        img_aucs.append(img_auc)
        pixel_aucs.append(pix_auc)
        pro_aucs.append(pro_auc)
        
    for i, class_name in enumerate(CLASS_NAMES):
        print(f'{class_name}: Image-AUC: {img_aucs[i]}, Pixel-AUC: {pixel_aucs[i]}, Pro-AUC: {pro_aucs[i]}')
    print('Mean Image-AUC: {}'.format(np.mean(img_aucs)))
    print('Mean Pixel-AUC: {}'.format(np.mean(pixel_aucs)))
    print('Mean Pro-AUC: {}'.format(np.mean(pro_aucs)))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

