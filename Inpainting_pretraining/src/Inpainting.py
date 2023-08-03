import os

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader

from .dataset import Dataset
from .metrics import PSNR
from .models import InpaintingModel
from .utils import create_dir, save_sample_png
import torchvision


class Inpaint():
    def __init__(self, config):
        self.config = config

        self.debug = False
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
            print('test dataset:'.format(len(self.test_dataset)))
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, augment=False, training=False)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

            print('train dataset:{}'.format(len(self.train_dataset)))
            print('eval dataset:{}'.format(len(self.val_dataset)))

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

    def load(self):
        self.inpaint_model.load(self.config.MODEL_LOAD)

    def save(self):
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        max_up, max_down = 0, 0
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            for items in train_loader:
                self.inpaint_model.train()

                images, masks = self.cuda(*items)

                # train
                outputs_up, outputs_down, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged_up = (outputs_up * masks) + images * (1 - masks)
                outputs_merged_down = (outputs_down * masks) + images * (1 - masks)

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs


                # sample
                if iteration % self.config.TRAIN_SAMPLE_INTERVAL == 0:
                    img_list2 = [images * (1 - masks), images, outputs_merged_up, outputs_up, outputs_merged_down, outputs_down]
                    name_list2 = ['in', 'gt', 'up_p2', 'up_p1', 'd_p2', 'd_p1']
                    save_sample_png(sample_folder=self.config.TRAIN_SAMPLE_SAVE,
                                              sample_name='ite_{}_{}'.format(self.inpaint_model.iteration,
                                                                             0), img_list=img_list2,
                                              name_list=name_list2, pixel_max_cnt=255, height=-1,
                                              width=-1)


                # save model at checkpoints
                if iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                # evaluate model at checkpoints
                if iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    up, down = self.eval()
                    self.inpaint_model.iteration = iteration

                    state = False
                    if up > max_up:
                        max_up = up
                        state = True

                    if down > max_down:
                        max_down = down
                        state = True

                    if state:
                        self.save()
                        print('---increase-iteration:{}'.format(iteration))

                print(logs)

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )


        self.inpaint_model.eval()

        up_psnr_all = []
        down_psnr_all = []

        iteration = self.inpaint_model.iteration
        with torch.no_grad():
            for items in val_loader:
                images, masks = self.cuda(*items)

                # eval
                outputs_up, outputs_down, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged_up = (outputs_up * masks) + images * (1 - masks)
                outputs_merged_down = (outputs_down * masks) + images * (1 - masks)

                up_psnr, _= self.metric(images, outputs_up)
                up_psnr_all.append(up_psnr)

                down_psnr, _ = self.metric(images, outputs_down)
                down_psnr_all.append(down_psnr)


                # sample
                if len(up_psnr_all) % self.config.EVAL_SAMPLE_INTERVAL == 0:
                    img_list2 = [images * (1 - masks), images, outputs_merged_up, outputs_up, outputs_merged_down, outputs_down]
                    name_list2 = ['in', 'gt', 'up_p2', 'up_p1', 'd_p2', 'd_p1']
                    save_sample_png(sample_folder=self.config.EVAL_SAMPLE_SAVE,
                                          sample_name='ite_{}_{}'.format(iteration, len(up_psnr_all)), img_list=img_list2,
                                          name_list=name_list2, pixel_max_cnt=255, height=-1,
                                          width=-1)



                print(f'psnr: up:{up_psnr}/{np.average(up_psnr_all)}  down:{down_psnr}/{np.average(down_psnr_all)}  {len(up_psnr_all)}')

                if len(up_psnr_all) >= 1000:
                    break

            print(f'iteration:{iteration}  psnr_ave up:{np.average(up_psnr_all)}  down:{np.average(down_psnr_all)}')

            return np.average(up_psnr_all), np.average(down_psnr_all)

    def test(self):
        self.inpaint_model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []

        index = 0
        with torch.no_grad():
            for items in test_loader:
                images, masks = self.cuda(*items)
                index += 1

                outputs = self.inpaint_model(images, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))


                psnr, ssim = self.metric(images, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(), self.transf(images[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
                    lpips_list.append(pl)

                l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                l1_list.append(l1_loss)

                print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                                                                                ssim, np.average(ssim_list),
                                                                                l1_loss, np.average(l1_list),
                                                                                pl, np.average(lpips_list),
                                                                                len(ssim_list)))


                if len(ssim_list) % 1 == 0:
                    images_masked = images * (1 - masks)
                    img_list = [images_masked, images, outputs, outputs_merged]
                    name_list = ['in', 'gt', 'pre1', 'pre2']

                    save_sample_png(sample_folder=self.config.TEST_SAMPLE_SAVE, sample_name='{}_'.format(len(ssim_list)),
                                              img_list=img_list,
                                              name_list=name_list, pixel_max_cnt=255, height=-1, width=-1)

            print('psnr_ave:{} ssim_ave:{} l1_ave:{} lpips:{}'.format(np.average(psnr_list),
                                                                                 np.average(ssim_list),
                                                                                 np.average(l1_list),
                                                                                 np.average(lpips_list)))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim