import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch
from skimage.transform import resize  # se quiser testar resize em vez de crop (não obrigatório)

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


def center_crop(img, target_shape):
    h, w = img.shape[:2]
    th, tw = target_shape[:2]
    start_x = (w - tw) // 2
    start_y = (h - th) // 2
    return img[start_y:start_y+th, start_x:start_x+tw]


'''
# --------------------------------------------
# training code for DnCNN
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
#         https://github.com/cszn/DnCNN
#
# Reference:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'dncnn'
    testset_name = 'sidd'

    model_pool = 'model_zoo'
    results = 'results'
    result_name = testset_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name + '.pth')

    L_path = '/content/train/syntheticData/NOISY_SRGB/'
    H_path = '/content/train/syntheticData/GT_SRGB/'
    E_path = os.path.join(results, result_name)
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # Load model
    # ----------------------------------------

    from models.network_dncnn import DnCNN as net
    model = net(in_nc=3, out_nc=3, nc=64, nb=17, act_mode='BR')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info(f'Model path: {model_path}')
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info(f'Params number: {number_parameters}')

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path)

    logger.info(f'Model: {model_name} | Testset: {testset_name}')
    logger.info(f'Found {len(L_paths)} noisy images and {len(H_paths)} clean images')

    for idx, img in enumerate(L_paths):

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2single(img_L)
        img_L_tensor = util.single2tensor4(img_L).to(device)

        # Model inference
        img_E_tensor = model(img_L_tensor)
        img_E = util.tensor2uint(img_E_tensor)

        # Load ground truth
        img_H = util.imread_uint(H_paths[idx], n_channels=3)

        # Ensure same size (crop GT if needed)
        if img_E.shape != img_H.shape:
            logger.warning(f"[{img_name}] Shape mismatch: Estimated {img_E.shape}, GT {img_H.shape}. Applying center crop to GT.")
            img_H = center_crop(img_H, img_E.shape)

        # PSNR & SSIM
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info(f'{img_name+ext} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}')

        # Save result
        util.imsave(img_E, os.path.join(E_path, img_name + ext))

    # Average metrics
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info(f'Average PSNR/SSIM(RGB) - {result_name} - PSNR: {ave_psnr:.2f} dB; SSIM: {ave_ssim:.4f}')


if __name__ == '__main__':
    main()
