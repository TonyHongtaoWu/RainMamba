import numbers
import os.path as osp
import mmcv
from mmcv.runner import auto_fp16
from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
import torch
import torch.nn.functional as F
from torchvision.models.vgg import vgg16,vgg19
import torch.nn as nn
import random
import torchvision.transforms as transforms
from modules.Hilbert3d import Hilbert3d



@MODELS.register_module()
class Derainer(BaseModel):
    """Basic model for video deraining.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.patch_size = 16
        self.positive_range_initial = 0
        self.min_negative_distance_initial = 96
        self.num_samples = 10

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()

        for paramvgg in vgg_model.parameters():
            paramvgg.requires_grad = False

        self.PerceptualLoss = PerpetualLoss(vgg_model).cuda()

        self.ContrastLoss = ContrastLoss()
        self.total_train = 600000


    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq',))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)
        # print(lq.shape)
        # print(gt.shape)
        return self.forward_train(lq, gt)

    def hilbert_curve_large_scale(self, ):

        nf = 5
        H = 64
        W = 64

        hilbert_curve = list(
            Hilbert3d(width=H, height=W, depth=nf)) 
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

        return {
            'hilbert_curve_large_scale': hilbert_curve
        }

    def hilbert_curve_small_scale(self, ):


        nf = 5
        H = 32
        W = 32

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



 

    def sample_patches(self, input_tensor, gt, lq, patch_size=16, positive_range_initial=2, min_negative_distance_initial=64,
                       num_samples=1, overlap=False, step_counter_cl=0):

        B, T, C, H, W = input_tensor.shape
        anchors, positives, negatives = [], [], []
        decay_rate = 0.5
        progress_ratio = step_counter_cl / self.total_train
        min_negative_distance = int(max(min_negative_distance_initial * (decay_rate **progress_ratio), 60))
        positive_range = int(min(positive_range_initial + progress_ratio * (12 - positive_range_initial), 12))

        for b in range(B):
            for t in range(T):
                difference = torch.abs(gt[b, t] - lq[b, t])
                if difference.shape[0] == 3:
                    gray = difference.mean(dim=0)
                else:
                    gray = difference
                threshold = gray.mean()
                binary = (gray > threshold).float()
                points = torch.nonzero(binary, as_tuple=False)
                for _ in range(num_samples):
                    t_index = random.choice([t, max(0, t - 1), min(T - 1, t + 1)])
                    t_index_N = random.randint(0, T - 1)

                    if points.size(0) == 0:
                        anchor_x = random.randint(0, W - patch_size)
                        anchor_y = random.randint(0, H - patch_size)
                    else:
                        selected_point = points[random.randint(0, points.size(0) - 1)]
                        anchor_y = min(selected_point[0].item(), H - patch_size)
                        anchor_x = min(selected_point[1].item(), W - patch_size)
                    anchor = input_tensor[b, t, :, anchor_y:anchor_y + patch_size, anchor_x:anchor_x + patch_size]
                    anchors.append(anchor)

                    if overlap:
                        positive_x = random.randint(max(0, anchor_x - positive_range),
                                                    min(W - patch_size, anchor_x + positive_range))
                        positive_y = random.randint(max(0, anchor_y - positive_range),
                                                    min(H - patch_size, anchor_y + positive_range))
                    else:
                        xdirection = random.choice(['left', 'right'])
                        ydirection = random.choice(['up', 'down'])

                        distance = random.randint(1, positive_range)
                        if xdirection == 'left':
                            positive_x = max(0, anchor_x - patch_size - distance)
                        elif xdirection == 'right':
                            positive_x = min(W - patch_size, anchor_x + patch_size + distance)

                        if ydirection == 'up':
                            positive_y = max(0, anchor_y - patch_size - distance)
                        elif ydirection == 'down':  # 'down'
                            positive_y = min(H - patch_size, anchor_y + patch_size + distance)
                    positive = gt[b, t_index, :, positive_y:positive_y + patch_size,
                               positive_x:positive_x + patch_size]
                    positives.append(positive)

                    while True:
                        negative_x = random.randint(0, W - patch_size)
                        negative_y = random.randint(0, H - patch_size)
                        if (abs(negative_x - anchor_x) > min_negative_distance or
                                abs(negative_y - anchor_y) > min_negative_distance):
                            negative = lq[b, t_index_N, :, negative_y:negative_y + patch_size,
                                       negative_x:negative_x + patch_size]
                            
                            kernel_size = int(random.random() * 4.95)
                            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size                          
                            blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
                            color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)

                            if random.random() < 0.5:
                                negative = color_jitter(negative)
                            if random.random() < 0.5:
                                negative = blurring_image(negative)

                            negatives.append(negative)
                            break

        anchors = torch.stack(anchors).view(-1, C, patch_size, patch_size)
        positives = torch.stack(positives).view(-1, C, patch_size, patch_size)
        negatives = torch.stack(negatives).view(-1, C, patch_size, patch_size)

        return anchors, positives, negatives

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # result64 = self.hilbert_list64()
        # hilbert_curve64 = result64['hilbert_curve64'].to(device)

        # result128 = self.hilbert_list128()
        # hilbert_curve128 = result128['hilbert_curve128'].to(device)
        hilbert_curve_large_scale = torch.load('./Hilbert/hilbert_curve_large_scale.pt').to(device)
        hilbert_curve_small_scale = torch.load('./Hilbert/hilbert_curve_small_scale.pt').to(device)

        output = self.generator(lq, hilbert_curve_large_scale, hilbert_curve_small_scale)

        b, t, c, h, w = output.size()

        
        self.step_counter += 1

        anchors, positives, negatives = self.sample_patches(output, gt, lq, patch_size=self.patch_size,
                                                    positive_range_initial=self.positive_range_initial, min_negative_distance_initial=self.min_negative_distance_initial,
                                                    num_samples=int(self.num_samples/b), overlap=True, step_counter_cl= self.step_counter)



        loss_pix1 = self.pixel_loss(output.contiguous().view(b * t, c, h, w), gt.contiguous().view(b * t, c, h, w))
        loss_pix2 = 0.3 * self.PerceptualLoss(output.contiguous().view(b * t, c, h, w),
                                              gt.contiguous().view(b * t, c, h, w))
        loss_pix3 = 0.1 * torch.mean(
            self.ContrastLoss(anchors, positives, negatives))



        losses['loss_Pix'] = loss_pix1
        losses['loss_Percept'] = loss_pix2
        losses['loss_contra'] = loss_pix3

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output



class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        self.l1 = PerceptualLoss2()
        self.ab = ablation

    def forward(self, a, p, n):
        loss = 0
        d_ap, d_an = 0, 0
        d_ap = self.l1(a, p.detach())
        d_an = self.l1(a, n.detach())
        contrastive = d_ap / (d_an + 1e-7)
        loss = loss + contrastive
        return loss


class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


class PerceptualLoss2(nn.Module):
    def __init__(self):
        super(PerceptualLoss2, self).__init__()
        self.L1 = nn.L1Loss()
        vgg = vgg19(pretrained=True).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval()
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval()

    def forward(self, x, y):
        loss1 = self.L1(self.loss_net1(x), self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x), self.loss_net3(y))        
        loss = 0.5* loss1 + 0.5 * loss3 
        return loss



