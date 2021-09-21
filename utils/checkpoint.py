import logging
import os
from collections import OrderedDict
import logging
import torch



def align_and_update_state_dicts(model_state_dict, loaded_state_dict, silence = True):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1

    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        if not silence:
            logger.info(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    # use strict loading
    model.load_state_dict(model_state_dict)


def load_coco_resnet_101(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="backbone.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    # use strict loading
    model.load_state_dict(model_state_dict)


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,

    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        # pdb.set_trace()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))

        # import time
        # time.sleep(10)
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load_extra_data(self, f = None):
        checkpoint = {}
        if not f:
            self.logger.info("No checkpoint found. Initializing model from scratch")
        else:
            if 'e2e_mask_rcnn_R_50_FPN_1x.pth' in f:
                self.transfer_learning = True
                checkpoint['iteration'] = -1
        return checkpoint


    def load(self, f=None, test = False):
        if test:
            self.logger.info("Loading checkpoint from {}".format(f))

            checkpoint = self._load_file(f)

            load_state_dict(self.model, checkpoint.pop("model"))
        else:
            if 'e2e_mask_rcnn_R_50_FPN_1x.pth' in f:
                self.transfer_learning = True
            else:
                self.transfer_learning = False

            if  not self.transfer_learning and self.has_checkpoint():

                # override argument with existing checkpoint
                self.transfer_learning = False
                f = self.get_checkpoint_file()
                # f  = False
            if not f:
                # no checkpoint could be found
                self.logger.info("No checkpoint found. Initializing model from scratch")
                return {}
            self.logger.info("Loading checkpoint from {}".format(f))
            checkpoint = self._load_file(f)
            # delete this two because we add new module which transfer learning model does not have
            del checkpoint['scheduler'], checkpoint['optimizer']
            self._load_model(checkpoint)
            if self.transfer_learning:
                # default last epoch of loaded weight is 89999
                checkpoint['iteration'] = -1

            if not self.transfer_learning:
            # if use transfer learning , do not load pretrain model scheduler and optimizer
                if "optimizer" in checkpoint and self.optimizer:
                    self.logger.info("Loading optimizer from {}".format(f))

                    # pdb.set_trace()
                    # pdb.set_trace()
                    self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
                if "scheduler" in checkpoint and self.scheduler:
                    self.logger.info("Loading scheduler from {}".format(f))
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))




        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")

        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        # pdb.set_trace()
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        # pdb.set_trace()
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
            # pdb.set_trace()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""

        return last_saved.strip('\n')

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        # pdb.set_trace()
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        if not self.transfer_learning:
            load_state_dict(self.model, checkpoint.pop("model"))
        else:
            # pdb.set_trace()
            # delete roi_head.box/mask.predictor.cls_score/bbox_pred/mask_fcn_logits in state_dict
            pretrained_weights = checkpoint.pop("model")
            model_state_dict = self.model.state_dict()
            loaded_state_dict = strip_prefix_if_present(pretrained_weights , prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model_state_dict = {k:v for k,v in model_state_dict.items() if 'cls_score' not in k and 'bbox_pred' not in k
                                and 'mask_fcn_logits' not in k}
            self.model.load_state_dict(model_state_dict, strict= False)

