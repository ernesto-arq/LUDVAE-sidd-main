import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


def crop_to_match(img_a, img_b):
    
    # ----------------------------------------
    # Corta as duas imagens (centralmente) para o menor tamanho comum.
    # Garante que ambas tenham o mesmo shape no final.
    # ----------------------------------------
    
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    target_h = min(ha, hb)
    target_w = min(wa, wb)

    def center_crop(img, th, tw):
        h, w = img.shape[:2]
        top = (h - th) // 2
        left = (w - tw) // 2
        return img[top:top+th, left:left+tw]

    return center_crop(img_a, target_h, target_w), center_crop(img_b, target_h, target_w)


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'dncnn'
    testset_name = 'sidd'

    model_pool = 'model_zoo'
    results = 'results'
    result_name = f'{testset_name}_{model_name}'
    model_path = os.path.join(model_pool, model_name + '.pth')

    L_path = '/content/val_noisy/'
    H_path = '/content/val_clean/'

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

        # Ensure same size by cropping both
        if img_E.shape != img_H.shape:
            #logger.warning(f"[{img_name}] Shape mismatch: Estimated {img_E.shape}, GT {img_H.shape}. Cropping both.")
            img_E, img_H = crop_to_match(img_E, img_H)

        # Double-check shapes after crop
        if img_E.shape != img_H.shape:
            #logger.error(f"[{img_name}] Still mismatched after crop! Skipping. Shapes: {img_E.shape} vs {img_H.shape}")
            continue

        # PSNR & SSIM
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info(f'{img_name+ext} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}')

        # Save result
        util.imsave(img_E, os.path.join(E_path, img_name + ext))

    # Average metrics
    if test_results['psnr']:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info(f'Average PSNR/SSIM(RGB) - {result_name} - PSNR: {ave_psnr:.2f} dB; SSIM: {ave_ssim:.4f}')
    else:
        logger.warning("No valid image pairs evaluated due to shape mismatch.")


if __name__ == '__main__':
    main()
