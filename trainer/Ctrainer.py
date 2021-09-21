import time
import torch
import os
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import models
from base import BaseTrainer, DataPrefetcher
from collections import OrderedDict
from dataloaders import create_dataloader
from dataloaders.cityscapes import PseudoDataWithLabelPrefetcher
from dataloaders.transforms import BoxMaskGenerator
from utils import palette as palettes
from utils import transforms as local_transforms
import numpy as np
from utils.dist_utils import synchronize, concat_all_gather
from utils.helpers import colorize_mask
from utils.losses import KLDLoss, PixelContrastLoss
from utils.losses import abCE_loss
from utils.lr_scheduler import sigmoid_rampup
from utils.metrics import eval_metrics, AverageMeter
from utils.misc import generate_ignore_region_cutmix
import pdb


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    if config[name]['type'] == 'Adam':
        # remove extra params
        # del config[name]['args']['weight_decay']
        try:
            del config[name]['args']['momentum']
        except:
            pass
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class CTrainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, prefetch=True,
                 rank=-1, distributed=False, local_rank=-1, global_branch=False, args=None, ):
        super(CTrainer, self).__init__(model, loss, resume, config, train_loader, val_loader, rank=rank,
                                        distributed=distributed, local_rank=local_rank, args=args)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        self.num_classes = self.train_loader.dataset.num_classes
        self.global_branch = global_branch
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        self.rampup = config['trainer']['rampup']

        self.iters_per_epoch = config['lr_scheduler']['args']['iters_per_epoch']
        # if self.device == torch.device('cpu'): prefetch = False
        self.train_loader = DataPrefetcher(train_loader, device=self.device,
                                           stop_after=int(self.iters_per_epoch / self.args.world_size))
        self.val_loader = DataPrefetcher(val_loader, device=self.device)
        unlabel_loader = create_dataloader(config['unlabel_loader']['type'], **config['unlabel_loader']['args'])
        self.unlabel_loader = PseudoDataWithLabelPrefetcher(unlabel_loader, device=self.device,
                                                            stop_after=int(self.iters_per_epoch / self.args.world_size))

        # params for Mean Teacher framework
        self.student = self.model
        self.old_model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
        self.teacher = self.old_model = self._parallel_model(self.old_model)
        self.old_model.eval()
        self._copy_model(self.model, self.old_model)
        self.unlabel_start_epoch = config['trainer']['unlabel_start_epoch']
        self.start_ema_iter = (self.unlabel_start_epoch - 1) * self.iters_per_epoch
        self.consist_weight = config['trainer']['consist_weight']
        self.ema = config['trainer']['ema']  # True or False
        self.ema_alpha = config['trainer']['ema_alpha']  # 0.99 or 0.999 or 0.95
        self.use_cb_consist = config['trainer']['cb_threshold']
        if self.config['trainer']['cutmix']:
            self.cutmix_generator = BoxMaskGenerator()
        else:
            self.cutmix_generator = None


        if self.contrastive:
            if 'large'in self.config['name'] :
                FEAT_SIZE = {"BDD100K": (65, 65), "VOC": (41, 41), "CityScapes": (64, 128)}
            else:
                FEAT_SIZE = {"BDD100K": (65, 65), "VOC": (41, 41), "CityScapes": (33, 65)}

            self.feat_size = FEAT_SIZE[self.config['train_loader']['type']]
            # init contrastive teacher model, teacher do not need rec model
            contrastive_comp_t = self.init_contrastive_modules(config)
            self.embeddingModule_t = contrastive_comp_t['embed']
            self.embeddingModule_t = self._parallel_model(self.embeddingModule_t)
            self.contrastive_loss_weight = config['contrastive']['contrastive_loss_weight']
            self.contrastive_rampup = config['contrastive']['rampup']
            self.contrastive_start_epoch = self.config['contrastive']['start_epoch']
            self.start_sup_contrastive_iter = (self.contrastive_start_epoch - 1) * self.iters_per_epoch

            self.unsup_iter = 0
            self.CTLOSS = PixelContrastLoss(temperature=self.config['contrastive']['temperature'],
                                            neg_num=self.config['contrastive']['neg_num'],
                                            memory_bank=None,
                                            mining=not self.args.random_sample)
        if self.args.kld:
            self.kld_loss = KLDLoss()

    def _train_epoch(self, epoch):  # start from 1
        torch.autograd.set_detect_anomaly(True)
        if self.distributed:
            self.train_loader.set_epoch(epoch)
        self.logger.info('\n')
        self.model.train()
        # set BN
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                           torch.nn.parallel.DistributedDataParallel):
                self.model.module.freeze_bn()
                if not self.args.no_fix_teacher_bn:
                    self.teacher.module.eval()
            else:
                self.model.freeze_bn()
                if not self.args.no_fix_teacher_bn:
                    self.teacher.eval()

        self.wrt_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        if self.distributed:
            synchronize()
        curr_iter = (epoch - 1) * self.iters_per_epoch
        self._init_unlabel_canvas()
        # write cb threshold to tensorboard
        if curr_iter > self.start_ema_iter + self.args.world_size and self.cls_balance_threshold and self.rank <= 0:
            for cls in range(self.train_loader.loader.dataset.num_classes):
                self.writer.add_scalar(f'ema_cb/{cls}',
                                       self.unlabel_loader.loader.dataset.cls_conf_threshold[cls],
                                       epoch)
        # write learning rate
        if self.rank <= 0:
            for param_group in self.optimizer.param_groups:
                group_lr = param_group['lr']
                self.logger.info(f'group lr {group_lr}')

        if not (self.consist_weight > 0 and epoch >= self.unlabel_start_epoch):
            # pure supervised
            for batch_idx, (data, target) in enumerate(self.train_loader):
                LOSSES = {}
                curr_iter += self.args.world_size
                self.data_time.update(time.time() - tic)
                data, target = data.to(self.device), target.to(self.device)
                self.lr_scheduler.step(epoch=epoch - 1)
                # LOSS & OPTIMIZE
                self.optimizer.zero_grad()
                # update teacher with student weight from iter-1
                self._update_teacher(curr_iter)
                output = self.model(data)
                if self.args.contrastive:
                    feat, output = output
                    # sup contrastive loss
                    if epoch >= self.contrastive_start_epoch:
                        if isinstance(feat, OrderedDict):
                            feat = feat['layer4']
                        feat = self.forward_embedding(feat, self.embeddingModule, istrain=True)
                        predict = torch.nn.functional.interpolate(output,
                                                                  (feat.shape[2], feat.shape[3]), mode='bilinear')
                        _, predict = torch.max(predict, dim=1)
                        sup_contrast_loss = self.CTLOSS(feats=feat, labels=target, predict=predict)
                        LOSSES.update({'sup_c_loss': sup_contrast_loss})
                # sup ce loss
                loss = self.loss(output, target)
                LOSSES.update({'sup_loss': loss})
                total_loss = self._calculate_losses(LOSSES, curr_iter, epoch)
                total_loss.backward()
                self.optimizer.step()
                # measure elapsed time
                self.batch_time.update(time.time() - tic)
                tic = time.time()
                # LOGGING & TENSORBOARD
                if batch_idx % self.log_step == 0 and self.rank <= 0:
                    self.wrt_step = (epoch - 1) * self.iters_per_epoch + batch_idx * self.args.world_size
                    self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss.item(), self.wrt_step)
                    for key, value in LOSSES.items():
                        if value is not None:
                            self.writer.add_scalar(f'{self.wrt_mode}/{key}', value.item(), self.wrt_step)
                # FOR EVAL
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)
                synchronize()
                if self.rank <= 0 and batch_idx % self.log_step == 0:
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()
                    loss_info = 'TRAIN ({}) | Loss: {:.3f} Sup: {:.3f}'.format(epoch, self.total_loss.average,
                                                                               self.sup_loss.average)
                    if self.contrastive and epoch >= self.contrastive_start_epoch:
                        loss_info += ' Sup Contrast: {:.3f}'.format(self.sup_contrastive_loss.average)
                    metric_info = '|Acc {:.2f} mIoU {:.2f}'.format(pixAcc, mIoU, )
                    self.logger.info(loss_info + metric_info)
        else:

            loader_list = [self.train_loader, self.unlabel_loader]
            if self.cls_balance_threshold and epoch == self.unlabel_start_epoch:
                self.logger.info('init cb with threshold = 0.9 ')
                self.init_for_cb_weight()
            for batch_idx, data_pack in enumerate(zip(*loader_list)):
                data, target = data_pack[0]
                weak, strong, unlabel_target, idx = data_pack[1]
                LOSSES = {}
                curr_iter += self.args.world_size
                self.data_time.update(time.time() - tic)
                if isinstance(data, tuple):
                    weak, strong = data[0].to(self.device), data[1].to(self.device)
                    data = (weak, strong)
                else:
                    data = data.to(self.device)
                target = target.to(self.device)
                unlabel_target = unlabel_target.to(self.device)
                self.lr_scheduler.step(epoch=epoch - 1)
                # LOSS & OPTIMIZE
                self.optimizer.zero_grad()
                # update teacher with student weight from iter-1
                self._update_teacher(curr_iter)
                loss_dict, output = self._forward_semi_train(labeled_data=data,
                                                             labeled_target=target,
                                                             unlabeled_data_student=strong,
                                                             unlabeled_data_teacher=weak,
                                                             unlabeled_target=unlabel_target,
                                                             epoch=epoch )
                LOSSES.update(loss_dict)
                total_loss = self._calculate_losses(LOSSES, curr_iter, epoch)
                total_loss.backward()
                self.optimizer.step()
                # measure elapsed time
                self.batch_time.update(time.time() - tic)
                tic = time.time()
                # LOGGING & TENSORBOARD
                if batch_idx % self.log_step == 0 and self.rank <= 0:
                    self.wrt_step = (epoch - 1) * self.iters_per_epoch + batch_idx * self.args.world_size
                    self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss.item(), self.wrt_step)
                    for key, value in LOSSES.items():
                        if value is not None:
                            self.writer.add_scalar(f'{self.wrt_mode}/{key}', value.item(), self.wrt_step)
                # FOR EVAL
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)
                synchronize()
                if self.rank <= 0 and batch_idx % self.log_step == 0:
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()
                    loss_info = 'TRAIN ({}) | Loss: {:.3f} Sup: {:.3f}'.format(epoch, self.total_loss.average,
                                                                               self.sup_loss.average)
                    metric_info = '|Acc {:.2f} mIoU {:.2f}'.format(pixAcc, mIoU, )
                    if self.unlabel_start_epoch - 1 < epoch:
                        loss_info = loss_info + ' Consist_Loss: {:.3f}'.format(self.consist_loss.average)
                        if self.use_cb_consist:
                            loss_info = loss_info + ' WOCB: {:.3f}'.format(self.wocb_loss.average)
                        if self.args.kld:
                            loss_info = loss_info + ' KLD: {:.3f}'.format(self.kld.average)
                        if self.contrastive:
                            loss_info = loss_info + ' Sup Contrast: {:.3f}'.format(self.sup_contrastive_loss.average)
                    self.logger.info(loss_info + metric_info)
        if self.rank <= 0:
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            log = {'loss': self.total_loss.average,
                   **seg_metrics}
        else:
            log = {
                'loss': self.total_loss.average,
            }
        synchronize()
        return log

    def _forward_semi_train(self, labeled_data,
                            labeled_target,
                            unlabeled_data_student,
                            unlabeled_data_teacher,
                            unlabeled_target,
                            epoch):
        LOSS = {}
        num_labeled, num_unlabeled = labeled_data.shape[0], unlabeled_data_student.shape[0]
        teacher_data = unlabeled_data_teacher
        # prepare teacher
        with torch.no_grad():
            output = self.teacher(teacher_data)
            if self.contrastive:
                feats_t, teacher_seg = output
                feats_emb_t = self.forward_embedding(feats_t, self.embeddingModule_t, istrain=False)
            else:
                teacher_seg = output
        teacher_seg = F.softmax(teacher_seg, dim=1)
        if self.use_cb_consist:
            ori_softmax_result = teacher_seg.clone()
        # prepare student data
        total_num = num_unlabeled + num_labeled
        if self.args.cutmix_ablation:
            idx_shuffle = torch.range(0, total_num - 1).cuda().long()
        else:
            idx_shuffle = torch.randperm(total_num).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        student_data = torch.cat([labeled_data, unlabeled_data_student])
        student_data = student_data[idx_shuffle]
        # cutmix operation
        if self.cutmix_generator is not None:
            if self.args.cutmix_ablation: # use to do cutmix ablation study
                pred_seg = self._forward_cutmix_abl(self.student, student_data)
            else:
                pred_seg = self._forward_cutmix(self.student, student_data)
            if (self.contrastive and epoch >= self.contrastive_start_epoch):
                feats_s, pred_seg, ignore_mask = pred_seg
            else:
                pred_seg, ignore_mask = pred_seg
        else:
            pred_seg = self.student(student_data)
        # unshuffle pred and feat
        pred_seg = pred_seg[idx_unshuffle]
        pred_labeled_s = pred_seg[:num_labeled]
        pred_unlabeled_s = pred_seg[num_labeled:]
        ignore_mask_unlabeled = torch.ones(1).to(pred_labeled_s.device)
        ignore_mask_labeled = torch.ones(1).to(pred_labeled_s.device)
        if self.contrastive and epoch >= self.contrastive_start_epoch:
            for key, value in feats_s.items():
                feats_s[key] = value[idx_unshuffle]

        # supervised loss
        sup_loss = self.loss(pred_labeled_s, labeled_target, ignore_mask_labeled)
        LOSS.update({'sup_loss': sup_loss})
        # consist loss
        log_probs = F.log_softmax(pred_unlabeled_s.clone(), dim=1)
        ignore_mask_unlabeled = ignore_mask_unlabeled[:, None, ...]

        if self.use_cb_consist:
            # concate all the pred from different device to update the class-balanced threshold
            ori_softmax_result = concat_all_gather(ori_softmax_result)
            self.unlabel_loader.update_ema_cb_threshold(ori_softmax_result, ema=0.9,
                                                        gamma=self.config['trainer']["gamma"])
            mask = self.unlabel_loader.get_cb_threshould_mask(teacher_seg)
            ignore_mask_unlabeled = mask * ignore_mask_unlabeled
            if self.args.mask_unlabel:
                # this unlabel_mask is only use in the setting that all the data has label.(n_sup=-1)
                good_mask = unlabeled_target == teacher_seg.max(1)[1]
                ignore_mask_unlabeled = good_mask[:, None, ...].float() * ignore_mask_unlabeled

        if self.contrastive and self.canvas_number < 8 and self.rank <= 0 and \
                self.config['contrastive']['visualize']:
            self._update_unlabel_canvas(unlabeled_data_teacher, teacher_seg, pred_unlabeled_s)
        loss = (- teacher_seg * log_probs * ignore_mask_unlabeled).mean() \
               * ignore_mask_unlabeled.numel() / (ignore_mask_unlabeled.nonzero().size(0) + 1e-9)
        LOSS.update({'consist_loss': loss})
        if self.use_cb_consist:
            # only print for visual, not add in total_loss
            wo_cb_loss = (- teacher_seg * log_probs).mean()
            LOSS.update({'wo_cb_loss': wo_cb_loss})
        # KLD Regularization:
        if self.args.kld:
            kld_loss = self.kld_loss(pred_unlabeled_s)
            LOSS.update({'kld_loss': kld_loss})
        # contrastive loss
        if self.contrastive:
            feat_emb = self.forward_embedding(feats_s, self.embeddingModule)
            if epoch >= self.contrastive_start_epoch:
                feat_emb_label, feat_emb_unlabel = feat_emb[:num_labeled], feat_emb[num_labeled:]
                if not self.args.contrastive_cross_model and not self.args.contrastive_cross_set:
                    pred_labeled = torch.nn.functional.interpolate(pred_labeled_s,
                                                                   (feat_emb_label.shape[2], feat_emb_label.shape[3]),
                                                                   mode='bilinear')
                    _, pred_labeled = torch.max(pred_labeled, dim=1)
                    sup_contrast_loss = self.CTLOSS(feats=feat_emb_label, labels=labeled_target, predict=pred_labeled,
                                                   )
                elif self.args.contrastive_cross_set and epoch >= self.contrastive_start_epoch:
                    unlabeled_pred = torch.max(teacher_seg, dim=1)[1]
                    if self.use_cb_consist:
                        unlabeled_pred[(1 - ignore_mask_unlabeled[:, 0]) == 1] = 255
                    pred_seg = torch.nn.functional.interpolate(pred_seg.float(),
                                                               (feat_emb_label.shape[2], feat_emb_label.shape[3]),
                                                               mode='bilinear')
                    pred_seg = torch.max(pred_seg, dim=1)[1]
                    target = torch.cat([labeled_target, unlabeled_pred])
                    feat = torch.cat([feat_emb_label, feat_emb_unlabel])
                    feat_t = torch.cat([feat_emb_label, feats_emb_t])
                    if self.args.mask_contrast:
                        sup_contrast_loss = self.CTLOSS(feats=feat, feats_t=feat_t, labels=target, predict=pred_seg,
                                                        cb_mask=ignore_mask_unlabeled)
                    else:
                        sup_contrast_loss = self.CTLOSS(feats=feat, feats_t=feat_t, labels=target, predict=pred_seg )

                elif epoch >= self.contrastive_start_epoch:
                    unlabeled_pred = torch.max(teacher_seg, dim=1)[1]
                    unlabeled_pred[(1 - ignore_mask_unlabeled[:, 0]) == 1] = 255
                    unlabeled_pred = torch.nn.functional.interpolate(unlabeled_pred[:, None, ...].float(),
                                                                     (feat_emb_label.shape[2], feat_emb_label.shape[3]),
                                                                     mode='nearest')
                    unlabeled_pred_s = torch.nn.functional.interpolate(pred_unlabeled_s.float(),
                                                                       (feat_emb_label.shape[2],
                                                                        feat_emb_label.shape[3]),
                                                                       mode='nearest')

                    unlabeled_pred_s = torch.max(unlabeled_pred_s, dim=1)[1]
                    sup_contrast_loss = self.CTLOSS(feats=feat_emb_unlabel, feats_t=feats_emb_t,
                                                    labels=unlabeled_pred[:, 0, ...], predict=unlabeled_pred_s,
                                                    cb_mask=ignore_mask_unlabeled)
                LOSS.update({'sup_c_loss': sup_contrast_loss})
        # if self.contrastive and self.canvas_number < 8 and self.rank <= 0 and \
        #         self.config['contrastive']['visualize']:
        #     self._update_unlabel_canvas(unlabeled_data_teacher, unlabeled_teacher_seg, pred_unlabeled_s)
        return LOSS, pred_labeled_s

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        if self.rank <= 0 and epoch % 20:
            os.system('nvidia-smi')
        self._reset_metrics()
        curr_iter = epoch * self.iters_per_epoch
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                if self.contrastive:
                    feats, output = output
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                total_loss = loss
                self.sup_loss.update(loss.item())
                self.total_loss.update(total_loss.item())
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)
                # LIST OF IMAGE TO VIZ (30 images)
                if len(val_visual) < 30 and self.rank <= 0:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])
                # PRINT INFO
                if self.rank <= 0 and batch_idx % self.log_step == 0:
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()
                    loss_info = 'EVAL ({}) | Loss: {:.3f} Sup: {:.3f}'.format(epoch, self.total_loss.average,
                                                                              self.sup_loss.average)
                    metric_info = '|Acc {:.2f} mIoU {:.2f}'.format(pixAcc, mIoU, )
                    self.logger.info(loss_info + metric_info)
            synchronize()
            seg_metrics = self._get_seg_metrics()
            self.wrt_step = (epoch) * self.config['lr_scheduler']['args']['iters_per_epoch']
            if self.rank <= 0 and epoch > 30:
                # WRTING & VISUALIZING THE MASKS
                val_img = []
                palette = palettes.CityScpates_palette
                for d, t, o in val_visual:
                    d = self.restore_transform(d)
                    t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                    d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                    [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                    val_img.extend([d, t, o])
                val_img = torch.stack(val_img, 0)
                val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
                self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)
            # METRICS TO TENSORBOARD
            if self.rank <= 0:
                self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }
            synchronize()
        return log

    def _calculate_losses(self, LOSSES, curr_iter, epoch):
        total_loss = 0
        self.sup_loss.update(LOSSES['sup_loss'].item())
        total_loss += LOSSES['sup_loss']

        if self.contrastive and epoch >= self.contrastive_start_epoch:
            sup_contrast_loss = LOSSES['sup_c_loss'] * self.contrastive_loss_weight * \
                                sigmoid_rampup(curr_iter - (self.start_sup_contrastive_iter + 1),
                                               self.contrastive_rampup)
            self.sup_contrastive_loss.update(sup_contrast_loss.item())
            total_loss += sup_contrast_loss
        if epoch >= self.unlabel_start_epoch and 'consist_loss' in LOSSES.keys():
            consist_loss = LOSSES['consist_loss'] * sigmoid_rampup(
                curr_iter - (self.start_ema_iter + self.args.world_size), self.rampup) \
                           * self.consist_weight
            self.consist_loss.update(consist_loss.item())
            total_loss += consist_loss
            # only use this as comparison: cb loss
            if self.use_cb_consist:
                self.wocb_loss.update(LOSSES['wo_cb_loss'].item())
            if self.args.kld:
                kld_loss = LOSSES['kld_loss'] * sigmoid_rampup(curr_iter - (self.start_ema_iter + self.args.world_size),
                                                               self.rampup) * \
                           self.args.kld_weight
                self.kld.update(kld_loss.item())
                total_loss += kld_loss
        self.total_loss.update(total_loss.item())
        return total_loss

    def forward_embedding(self, feature, embed_module, istrain=True):
        if not istrain:
            with torch.no_grad():
                emb = embed_module(feature)
        else:
            emb = embed_module(feature)
        return emb

    def _forward_cutmix(self, model, img):
        batch, _, h, w = img.shape
        # mix
        cutmix_mask = self.cutmix_generator.generate_params(int(batch / 2), (h, w), rng=None)
        cutmix_mask = torch.from_numpy(cutmix_mask).float().to(self.device)
        img_1 = img[:int(batch / 2)] * (1 - cutmix_mask) + img[int(batch / 2):] * (cutmix_mask)
        img_2 = img[:int(batch / 2)] * (cutmix_mask) + img[int(batch / 2):] * (1 - cutmix_mask)
        pred = model(torch.cat([img_1, img_2]))
        if self.contrastive :
            feats, pred = pred
            if self.config['train_loader']['type'] == 'CityScapes':
                id_to_name = {0: 'layer1', 1: 'layer2', 2: 'layer3', 3: 'layer4'}
                feat_size = [(65, 129), (33, 65), (33, 65), (33, 65)]
            elif self.config['train_loader']['type'] == 'BDD100K':
                id_to_name = {0: 'layer1', 1: 'layer2', 2: 'layer3', 3: 'layer4'}
                feat_size = [(129, 129), (65, 65), (65, 65), (65, 65)]

            for idx, resolution in enumerate(feat_size):
                if idx == 0: continue
                _feat = feats[id_to_name[idx]]
                cutmix_mask_feat = F.interpolate(cutmix_mask, self.feat_size)
                feats_1 = _feat[:int(batch / 2)] * (1 - cutmix_mask_feat) + _feat[int(batch / 2):] * cutmix_mask_feat
                feats_2 = _feat[:int(batch / 2)] * (cutmix_mask_feat) + _feat[int(batch / 2):] * (1 - cutmix_mask_feat)
                feats[id_to_name[idx]] = torch.cat([feats_1, feats_2])
        # unmix
        pred_1 = pred[:int(batch / 2)] * (1 - cutmix_mask) + pred[int(batch / 2):] * cutmix_mask
        pred_2 = pred[:int(batch / 2)] * (cutmix_mask) + pred[int(batch / 2):] * (1 - cutmix_mask)
        pred_seg = torch.cat([pred_1, pred_2])
        ignore_mask = generate_ignore_region_cutmix(cutmix_mask)
        ignore_mask = ignore_mask.to(pred_seg.device)
        ignore_mask = torch.cat([ignore_mask, ignore_mask], dim=0)
        if self.contrastive:
            return feats, pred_seg, ignore_mask
        else:
            return pred_seg, ignore_mask

    def _forward_cutmix_abl(self, model, img):
        batch, _, h, w = img.shape
        # mix
        cutmix_mask = self.cutmix_generator.generate_params(int(batch / 2), (h, w), rng=None)
        cutmix_mask = torch.from_numpy(cutmix_mask).float().to(self.device)
        img_1 = img[:int(batch / 4)] * (1 - cutmix_mask[:int(batch / 4)]) + img[int(batch / 4):int(batch / 2)] * (
        cutmix_mask[:int(batch / 4)])
        img_2 = img[:int(batch / 4)] * (cutmix_mask[:int(batch / 4)]) + img[int(batch / 4):int(batch / 2)] * (
                    1 - cutmix_mask[:int(batch / 4)])
        img_3 = img[int(batch / 2):int(3 * batch / 4)] * (1 - cutmix_mask[int(batch / 4):]) + img[
                                                                                              int(3 * batch / 4):] * (
                    cutmix_mask[int(batch / 4):])
        img_4 = img[int(batch / 2):int(3 * batch / 4)] * (cutmix_mask[int(batch / 4):]) + img[int(3 * batch / 4):] * (
                1 - cutmix_mask[int(batch / 4):])
        pred = model(torch.cat([img_1, img_2, img_3, img_4]))
        # unmix
        pred_1 = pred[:int(batch / 4)] * (1 - cutmix_mask[:int(batch / 4)]) + pred[int(batch / 4):int(batch / 2)] * (
        cutmix_mask[:int(batch / 4)])
        pred_2 = pred[:int(batch / 4)] * (cutmix_mask[:int(batch / 4)]) + pred[int(batch / 4):int(batch / 2)] * (
                    1 - cutmix_mask[:int(batch / 4)])
        pred_3 = pred[int(batch / 2):int(3 * batch / 4)] * (1 - cutmix_mask[int(batch / 4):]) + pred[
                                                                                                int(3 * batch / 4):] * (
                     cutmix_mask[int(batch / 4):])
        pred_4 = pred[int(batch / 2):int(3 * batch / 4)] * (cutmix_mask[int(batch / 4):]) + pred[
                                                                                            int(3 * batch / 4):] * (
                         1 - cutmix_mask[int(batch / 4):])
        pred_seg = torch.cat([pred_1, pred_2, pred_3, pred_4])
        ignore_mask = generate_ignore_region_cutmix(cutmix_mask)
        ignore_mask = ignore_mask.to(pred_seg.device)
        ignore_mask = torch.cat([ignore_mask, ignore_mask], dim=0)
        return pred_seg, ignore_mask

    def init_for_cb_weight(self):
        self.unlabel_loader.init_cb_threshold(prob=0.9)

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.sup_loss = AverageMeter()
        self.consist_loss = AverageMeter()
        self.wocb_loss = AverageMeter()
        self.sup_contrastive_loss = AverageMeter()
        self.kld = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        self.total_f1 = 0
        self.total_iteration = 0

    def _updata_cls_metrics(self, f1):
        self.total_f1 += f1
        self.total_iteration += 1

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    def _get_cls_metrics(self):
        return np.round(self.total_f1 / self.total_iteration, 4)

    def _copy_model(self, fromthismodel, tothismodel):
        # also copy bn buffer
        mp = list(fromthismodel.parameters())
        mcp = list(tothismodel.parameters())
        n = len(mp)
        for i in range(n):
            mcp[i].data[:] = mp[i].data[:]
        for mp_buffer, mcp_buffer in zip(*[fromthismodel.named_buffers(), tothismodel.named_buffers()]):
            if mp_buffer[0] == mcp_buffer[0] and 'running' in mcp_buffer[0]:
                mcp_buffer[1].data[:] = mp_buffer[1].data[:].clone()

    def _update_teacher(self, iter):
        if self.ema:
            if iter == self.start_ema_iter + self.args.world_size:
                self._copy_model(self.student, self.teacher)
                if self.contrastive:
                    self._copy_model(self.embeddingModule, self.embeddingModule_t)

            elif iter < self.start_ema_iter + self.args.world_size:
                pass
            else:
                times = iter - self.start_ema_iter
                alpha = min(1 - 1 / (times + 1), self.ema_alpha)
                self._ema_(self.teacher, self.student, alpha)
                if self.contrastive:
                    self._ema_(self.embeddingModule_t, self.embeddingModule, alpha)
        else:
            self._copy_model(self.student, self.teacher)
            if self.contrastive:
                self._copy_model(self.embeddingModule, self.embeddingModule_t)

    def _ema_(self, teacher, student, alpha):
        for ema_param, param in zip(teacher.parameters(),
                                    student.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _init_unlabel_canvas(self):
        self.unlabel_canvas = []
        self.canvas_number = 0

    def _update_unlabel_canvas(self, image, predict_t, predict_s, ):
        output_t = predict_t.data.max(1)[1].cpu().numpy()
        pred_t = predict_t.data.max(1)[0].cpu().numpy()
        output_s = predict_s.data.max(1)[1].cpu().numpy()
        confident_predict = predict_t.max(1)[1].cpu()
        confident_predict = confident_predict.numpy()
        confident_predict[pred_t < self.config['contrastive']['confidence']] = 255
        image = image.data.cpu()
        self.unlabel_canvas.append([image[0], output_t[0], output_s[0], confident_predict[0]])
        self.canvas_number += 1
