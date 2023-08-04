import os

import lpips
import numpy as np
import scipy
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from scipy.misc import imread
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

transf = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()


def load_item(gt_path, pre_path, mask_path):

    gt = imread(gt_path)
    pre = imread(pre_path)
    mask = imread(mask_path)

    gt = resize(gt)
    pre = resize(pre)
    mask = resize(mask)

    mask = (mask > 255 * 0.9).astype(np.uint8) * 255

    return to_tensor(gt), to_tensor(pre), to_tensor(mask)


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    img_t = img_t.unsqueeze(dim=0)
    return img_t


def resize(img):
    img = scipy.misc.imresize(img, [256, 256], 'cubic')
    return img


def metric(gt, pre):
    lpips = loss_fn_vgg(transf(pre[0]).cuda(), transf(gt[0]).cuda()).item()

    pre = pre * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

    gt = gt * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

    psnr = compare_psnr(gt, pre)
    ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

    return psnr, ssim, lpips


def evaluation(gt_root, pre_root, mask_root):
    fnames = os.listdir(gt_root)

    fnames.sort()

    psnr_all_list, ssim_all_list, lpips_all_list = [], [], []
    psnr_non_list, ssim_non_list, lpips_non_list = [], [], []
    psnr_shadow_list, ssim_shadow_list, lpips_shadow_list = [], [], []

    for fname in fnames:
        gt_path = os.path.join(gt_root, fname)
        pre_path = os.path.join(pre_root, fname)
        mask_path = os.path.join(mask_root, fname)

        gt, pre, mask = load_item(gt_path, pre_path, mask_path)

        psnr_all, ssim_all, lpips_all = metric(gt, pre)
        psnr_non, ssim_non, lpips_non = metric(gt*(1-mask), pre*(1-mask))
        psnr_shadow, ssim_shadow, lpips_shadow = metric(gt*mask, pre*mask)

        psnr_all_list.append(psnr_all)
        ssim_all_list.append(ssim_all)
        lpips_all_list.append(lpips_all)

        psnr_non_list.append(psnr_non)
        ssim_non_list.append(ssim_non)
        lpips_non_list.append(lpips_non)

        psnr_shadow_list.append(psnr_shadow)
        ssim_shadow_list.append(ssim_shadow)
        lpips_shadow_list.append(lpips_shadow)

        print(f'ALL psnr: {round(psnr_all,4)}/{round(np.average(psnr_all_list), 4)}  '
              f'ssim: {round(ssim_all, 4)}/{round(np.average(ssim_all_list), 4)}  '
              f'lpips:{round(lpips_all, 4)}/{round(np.average(lpips_all_list), 4)} | '
              
              f'Shadow psnr:{round(psnr_shadow, 4)}/{round(np.average(psnr_shadow_list), 4)}  '
              f'ssim:{round(ssim_shadow, 4)}/{round(np.average(ssim_shadow_list), 4)}  '
              f'lipis:{round(lpips_shadow, 4)}/{round(np.average(lpips_shadow_list), 4)} | '
              
              f'Non psnr:{round(psnr_non, 4)}/{round(np.average(psnr_non_list), 4)}  '
              f'ssim:{round(ssim_non, 4)}/{round(np.average(ssim_non_list), 4)}  '
              f'lipis:{round(lpips_non, 4)}/{round(np.average(lpips_non_list), 4)}  '
              
              f'    {len(psnr_all_list)}')

    print('-----------------------------------------------------------------------------')
    print(f'All psnr:{round(np.average(psnr_all_list), 4)} ssim:{round(np.average(ssim_all_list), 4)} lpips:{round(np.average(lpips_all_list), 4)}')
    print(f'Shadow psnr:{round(np.average(psnr_shadow_list), 4)} ssim:{round(np.average(ssim_shadow_list), 4)} lpips:{round(np.average(lpips_shadow_list), 4)}')
    print(f'Non psnr:{round(np.average(psnr_non_list), 4)} ssim:{round(np.average(ssim_non_list), 4)} lpips:{round(np.average(lpips_non_list), 4)}')

mask_root = 'test/test_B'
gt_root = 'test/test_C_fixed'
ICCV_ours = 'result/pre'

evaluation(gt_root, ICCV_ours, mask_root)


