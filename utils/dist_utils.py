import os
import subprocess
import numpy as np
import pickle
import torch
import functools
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import logging

def init_dist(launcher, args, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        init_dist_pytorch(args, backend)
    elif launcher == 'slurm':
        init_dist_slurm(args, backend)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))

def init_dist_pytorch(args, backend="nccl"):
    args.rank = int(os.environ['LOCAL_RANK'])
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.rank
    args.world_size = args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend=backend)

def init_dist_slurm(args, backend="nccl"):
    args.rank = int(os.environ['SLURM_PROCID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu = args.rank % args.ngpus_per_node
    torch.cuda.set_device(args.gpu)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(args.tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    dist.init_process_group(backend=backend)
    args.total_gpus = dist.get_world_size()

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(rank_list[i]))
    group_size = world_size // num_groups
    print ("Rank no.{} start sync BN on the process group of {}".format(rank, rank_list[rank//group_size]))
    return groups[rank//group_size]

def convert_sync_bn(model, process_group=None, gpu=None):
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            if (gpu is not None):
                m = m.cuda(gpu)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group, gpu)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def print_dist_information():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f'-----------------------------------\n'
        f'Distributed training ENV: \n'
          f'rank    {rank}\n'
          f'world_size    {world_size}\n'
        f'-----------------------------------\n')

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def _serialize_to_tensor(data, group, device = None):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    if device is None:
        device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)

    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def broadcast_value(value, device, src):
    tensor = torch.tensor(value).cuda(device)
    dist.broadcast(tensor, src)
    value = tensor.item()
    return value


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []

@torch.no_grad()
def all_gather(data, group=None, device = None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()

    if dist.get_world_size(group) == 1:
        return [data]
    tensor = _serialize_to_tensor(data, group, device)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

@torch.no_grad()
def all_gather_tensor(data, group=None, device = None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()

    if dist.get_world_size(group) == 1:
        return [data]
    tensor = _serialize_to_tensor(data, group, device)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list



@torch.no_grad()
def concat_all_gather(tensor, dim = 0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_world_size() ==1:
        return tensor

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=dim)
    return output

def reduce_tensor(tensor):
    return tensor
    rt = tensor.clone()
    if get_world_size() > 1:
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= dist.get_world_size()
    return rt