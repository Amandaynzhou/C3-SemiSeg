import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler

from base.samplers import IterDistributedSampler

# from multiprocessing.util



class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split=0.0, distributed=False,
                 world_size=1, rank=-1, num_samples=100, collect_fn=None, labeled_ratio=1.):
        # num sample only work at shuffle mode
        self.shuffle = shuffle
        self.dataset = dataset
        self.distributed = distributed
        if not self.distributed:
            self.nbr_examples = len(dataset)
            if self.shuffle:
                self.train_sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
                self.shuffle = False  # use random sampler instead of shuffle = True in train phase
            else:
                self.train_sampler, self.val_sampler = None, None
        else:
            if self.dataset.split != 'train':
                self.train_sampler = None
            else:
                if not self.shuffle:
                    self.train_sampler = None
                else:
                    self.train_sampler = IterDistributedSampler(
                        self.dataset, num_replicas=world_size, rank=rank, num_samples=num_samples)
                self.shuffle = False
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'collate_fn': collect_fn
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        # self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)

    def get_cb_threshould_mask(self, prob):
        return self.dataset.get_cb_threshould_mask(prob)


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch):
        if self.loader.distributed:
            # only distributed loader need to set epoch
            self.loader.sampler.set_epoch(epoch)
        else:
            try:
                self.loader.sampler.set_epoch(epoch)
            except:
                pass

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
