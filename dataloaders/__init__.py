
from .cityscapes import CityScapes, PseudoCityScapes
from .bdd import BDD100K,PseudoBDD100K

__factory = {

             'CityScapes': CityScapes,
             'PseudoCityScapes':PseudoCityScapes,
             'BDD100K': BDD100K,
             'PseudoBDD100K': PseudoBDD100K,
            }

def names():
    return sorted(__factory.keys())

def create_dataloader(name, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'pitts', 'tokyo'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown loader:", name)
    return __factory[name]( *args, **kwargs)

from .cityscapes import CityScapes, PseudoCityScapes
from .bdd import BDD100K,PseudoBDD100K

__factory = {

             'CityScapes': CityScapes,
             'PseudoCityScapes':PseudoCityScapes,
             'BDD100K': BDD100K,
             'PseudoBDD100K': PseudoBDD100K,
            }

def names():
    return sorted(__factory.keys())

def create_dataloader(name, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'pitts', 'tokyo'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown loader:", name)
    return __factory[name]( *args, **kwargs)
