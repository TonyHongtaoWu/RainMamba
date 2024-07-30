# https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/restorers/basic_restorer.py
# https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/restorers/basicvsr.py
import numbers
import os.path as osp
import mmcv
import numpy as np
import torch
from mmedit.core import tensor2img
from ..registry import MODELS
from .derainer import Derainer
from modules.Hilbert3d import Hilbert3d

@MODELS.register_module()
class DrainNet(Derainer):
    """Basic model for video deraining.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        ensemble (dict): Config for ensemble. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """


    def __init__(self,
                 generator,
                 pixel_loss,
                 ensemble=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        # ensemble
        self.forward_ensemble = None
        if ensemble is not None:
            if ensemble['type'] == 'SpatialTemporalEnsemble':
                from mmedit.models.common.ensemble import \
                    SpatialTemporalEnsemble
                is_temporal = ensemble.get('is_temporal_ensemble', False)
                self.forward_ensemble = SpatialTemporalEnsemble(is_temporal)
            else:
                raise NotImplementedError(
                    'Currently support only '
                    '"SpatialTemporalEnsemble", but got type '
                    f'[{ensemble["type"]}]')

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)

        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2img(output[:, i, :, :, :])
                    gt_i = tensor2img(gt[:, i, :, :, :])
                    avg.append(self.allowed_metrics[metric](
                        output_i, gt_i, crop_border, convert_to=convert_to))
                eval_result[metric] = np.mean(avg)
            elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
                output_img = tensor2img(output)
                gt_img = tensor2img(gt)
                value = self.allowed_metrics[metric](
                    output_img, gt_img, crop_border, convert_to=convert_to)
                eval_result[metric] = value

        return eval_result



    def hilbert_curve_large_scale(self, ):

        # B, nf, C, H, W = x.shape

        nf = 5
        H = 32
        W = 32

        hilbert_curve = list(
            Hilbert3d(width=H, height=W, depth=nf))  
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

        return {
            'hilbert_curve_large_scale': hilbert_curve
        }

    def hilbert_curve_small_scale(self, ):

        # B, nf, C, H, W = x.shape

        nf = 5
        H = 16
        W = 16

        hilbert_curve = list(
            Hilbert3d(width=H, height=W, depth=nf)) 
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

        return {
            'hilbert_curve_small_scale': hilbert_curve
        }
    
        
        
    def save_hilbert_curve_large_scale(self, hilbert_curve_large_scale, filename='./Hilbert/hilbert_curve_large_scale.pt'):
        torch.save(hilbert_curve_large_scale, filename)
        
    def save_hilbert_curve_small_scale(self, hilbert_curve_small_scale, filename='./Hilbert/hilbert_curve_small_scale.pt'):
        torch.save(hilbert_curve_small_scale, filename)

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        with torch.no_grad():
            if self.forward_ensemble is not None:
                output = self.forward_ensemble(lq, self.generator)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # result64 = self.hilbert_curve_large_scale()
                # hilbert_curve64 = result64['hilbert_curve_large_scale'].to(device)
                
                # self.save_hilbert_curve_large_scale(hilbert_curve64)

                # result128 = self.hilbert_curve_small_scale()
                # hilbert_curve128 = result128['hilbert_curve_small_scale'].to(device)
                
                # self.save_hilbert_curve_small_scale(hilbert_curve64)
                
                
                
                hilbert_curve_large_scale = torch.load('./Hilbert/hilbert_curve_large_scale.pt').to(device)
                hilbert_curve_small_scale = torch.load('./Hilbert/hilbert_curve_small_scale.pt').to(device)
                
                output = test_video(lq, self.generator, hilbert_curve_large_scale, hilbert_curve_small_scale)




        if gt is not None and gt.ndim == 4:
            t = output.size(1)
            if self.check_if_mirror_extended(lq):  # with mirror extension
                output = 0.5 * (output[:, t // 4] + output[:, -1 - t // 4])
            else:  # without mirror extension
                output = output[:, t // 2]

        results = dict(lq=lq.cpu(), output=output.cpu())
        # save image
        if save_image:
            if output.ndim == 4:  # an image, key = 000001/0000 (Vimeo-90K)
                img_name = meta[0]['key'].replace('/', '_')
                if isinstance(iteration, numbers.Number):
                    save_path = osp.join(
                        save_path, f'{img_name}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path = osp.join(save_path, f'{img_name}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), save_path)
            elif output.ndim == 5:  # a sequence, key = 000
                folder_name = meta[0]['key'].split('/')[0]
                for i in range(0, output.size(1)):
                    if isinstance(iteration, numbers.Number):
                        save_path_i = osp.join(
                            save_path, folder_name,
                            f'{i:08d}-{iteration + 1:06d}.png')
                    elif iteration is None:
                        save_path_i = osp.join(save_path, folder_name,
                                               f'{i:04d}.png')
                    else:
                        raise ValueError('iteration should be number or None, '
                                         f'but got {type(iteration)}')
                    mmcv.imwrite(
                        tensor2img(output[:, i, :, :, :]), save_path_i)

        return results



def test_video(lq, model, hilbert_curve_large_scale, hilbert_curve_small_scale):
        '''test the video as a whole or as clips (divided temporally). '''


        tile = [5, 256, 256]
        tile_overlap = [1, 4, 4]
        scale = 1
        window_size = [2, 8, 8]
        nonblind_denoising = False

        num_frame_testing = tile[0]
        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = scale
            num_frame_overlapping = tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, hilbert_curve_large_scale, hilbert_curve_small_scale)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = window_size
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model)
            output = output[:, :d_old, :, :, :]

        return output

def test_clip(lq, model, hilbert_curve_large_scale, hilbert_curve_small_scale):
    ''' test the clip as a whole or as patches. '''

    tile = [5, 256, 256]
    tile_overlap = [1, 4, 4]
    scale = 1
    window_size = [2, 8, 8]
    nonblind_denoising = False
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # result64 = hilbert_list64()
    # hilbert_curve64 = result64['hilbert_curve64'].to(device)

    # result128 = hilbert_list128()
    # hilbert_curve128 = result128['hilbert_curve128'].to(device)
        
        

    sf = scale
    window_size = window_size
    size_patch_testing = tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch = model(in_patch, hilbert_curve_large_scale, hilbert_curve_small_scale).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq, hilbert_curve_large_scale, hilbert_curve_small_scale).detach().cpu()

        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output
