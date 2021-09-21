import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dist_utils import all_gather
from utils.lovasz_losses import lovasz_softmax


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class MaskCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(MaskCrossEntropyLoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, mask=None):
        loss = self.CE(output, target)
        ignore_index_mask = (target != self.ignore_index).float()
        if mask is None:
            mask = ignore_index_mask
        else:
            mask = mask * ignore_index_mask
        if mask.nonzero().size(0) != 0:
            loss = (loss * mask).mean() * mask.numel() / mask.nonzero().size(0)
        else:
            loss = (loss * mask).mean() * mask.numel() / (mask.nonzero().size(0) + 1e-9)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        logpt = logpt.clamp(-1e-6, 1e6)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            mask = (target != self.ignore_index).long()[:, None, ...]
            result = (loss * mask).sum() / mask.nonzero().size(0)
            return result
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class EntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(EntropyLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss
        self.ignore_index = ignore_index

    def forward(self, output, target):
        b = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        b = b * (target != self.ignore_index).long()[:, None, ...]
        b = -1.0 * b.sum() / (b.nonzero().size(0))
        return b


class KLDLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(KLDLoss, self).__init__()

        self.ignore_index = ignore_index

    def forward(self, output, target=None):
        b = F.log_softmax(output, dim=1)
        if target is not None:
            b = b * (target == self.ignore_index).long()[:, None, ...]
        b = -1.0 * b.sum() / ((b.nonzero().size(0)))
        return b


def sharpen(p, temp=0.5):
    pt = p ** (1 / temp)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    targets_u = targets_u.detach()
    return targets_u


def label_smooth(onehot, cls=19, eta=0.1):
    low_confidence = eta / cls
    new_label = (1 - eta) * onehot + low_confidence
    return new_label


class PixelContrastLoss(nn.Module):
    def __init__(self, temperature=0.1, neg_num=256, memory_bank=None, mining=True):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = 0.1
        self.ignore_label = 255
        self.max_samples = int(neg_num * 19)
        self.max_views = neg_num
        self.memory_bank = memory_bank
        self.mining = mining

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        # filter each image, to find what class they have num > self.max_view pixel
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)
        n_view = self.max_views
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            # hard: predict wrong
            for cls_id in this_classes:
                if self.mining:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                        raise Exception

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1
                else:
                    num_indice = (this_y_hat == cls_id).nonzero()
                    number = num_indice.shape[0]
                    perm = torch.randperm(number)
                    indices = num_indice[perm[:n_view]]
                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1
        return X_, y_

    def _hard_pair_sample_mining(self, X, X_2, y_hat, y, mask=None):

        batch_size, feat_dim = X.shape[0], X.shape[-1]
        if mask is not None:
            y_hat = mask * y_hat + (1 - mask) * 255
        classes = []
        total_classes = 0
        # filter each image, to find what class they have num > self.max_view pixel
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x > 0 and x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)
        n_view = self.max_views
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        X_2_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            # hard: predict wrong
            for cls_id in this_classes:
                if self.mining:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                        raise Exception

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    X_2_[X_ptr, :, :] = X_2[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1
                else:

                    num_indice = (this_y_hat == cls_id).nonzero()
                    number = num_indice.shape[0]
                    perm = torch.randperm(number)
                    indices = num_indice[perm[:n_view]]
                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    X_2_[X_ptr, :, :] = X_2[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1

        return X_, X_2_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        # the reason for use unbind is that they need to repeat mask n_view times later
        # 对于每个class的样本，它有(n_view - 1)* ptr中有cls的图片的个数 个 正样本
        # 有 n_view * (ptr- cls的次数)个负样本
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        # max是自身
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # set self = 0
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _contrastive_pair(self, feats_, feats_t, labels_, labels_t):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        anchor_num_t, n_view_t = feats_t.shape[0], feats_t.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)
        labels_t = labels_t.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_t, 0, 1)).float().cuda()

        contrast_count = n_view_t
        # the reason for use unbind is that they need to repeat mask n_view times later
        # 对于每个class的样本，它有(n_view - 1)* ptr中有cls的图片的个数 个 正样本
        # 有 n_view * (ptr- cls的次数)个负样本
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        contrast_feature_t = torch.cat(torch.unbind(feats_t, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = n_view

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature_t.detach(), 0, 1)),
            self.temperature)
        # max是自身
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        # set self = 0
        if anchor_num == anchor_num_t:
            logits_mask = torch.ones_like(mask).scatter_(1,
                                                         torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                         0)
            mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, feats_t=None, labels=None, predict=None, cb_mask=None):
        # feat from student, feats_2 from teacher if have

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        if cb_mask is not None:
            mask_batch = cb_mask.shape[0]
            batch, _, h, w = feats.shape
            cb_mask = F.interpolate(cb_mask.float(), (h, w), mode='nearest')
            if mask_batch != batch: # when use both labeled and unlabeld data for loss, labeled do not need any ignore
                labeled_confidence_mask = torch.ones_like(cb_mask).to(cb_mask.dtype)
                cb_mask = torch.cat([labeled_confidence_mask, cb_mask])
            cb_mask = cb_mask.view(batch, -1)
        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats = F.normalize(feats, dim=2)
        if feats_t is None:
            feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
            if feats_ is None:
                print('input do not have enough label, ignore (happened in beginning of BDD)')
                loss = 0 * feats.mean()
                return loss

            if self.memory_bank is not None:
                self.memory_bank.dequeue_and_enqueue(feats_, labels_)
                feats_t_, labels_t_ = self.memory_bank.get_valid_feat_and_label()

                loss = self._contrastive_pair(feats_, feats_t_, labels_, labels_t_)
            else:
                loss = self._contrastive(feats_, labels_)
        else:

            feats_t = feats_t.permute(0, 2, 3, 1)
            feats_t = feats_t.contiguous().view(feats_t.shape[0], -1, feats_t.shape[-1])
            feats_t = F.normalize(feats_t, dim=2)
            feats_, feats_t_, labels_ = self._hard_pair_sample_mining(feats, feats_t, labels, predict, cb_mask)
            if feats_ is None:
                print('input do not have enough label, ignore (happened in beginning of BDD)')
                loss = 0 * feats.mean()
                return loss

            if self.memory_bank is not None:

                self.memory_bank.dequeue_and_enqueue(feats_t_, labels_)
                feats_t_, labels_t_ = self.memory_bank.get_valid_feat_and_label()

                loss = self._contrastive_pair(feats_, feats_t_, labels_, labels_t_)
            else:
                loss = self._contrastive_pair(feats_, feats_t_, labels_, labels_)
        return loss


def log_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1 - np.exp(-5.0 * current / rampup_length))


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """

    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                 reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = log_rampup
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter):
        cur_total_iter = curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter):

        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter)
        else:
            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')
