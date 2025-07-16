from .abstract_model_interface import AbstractSegment, AbstractModel, AbstractModelNoMesh
from .biorbd_model_interface import BiorbdModelNoMesh, BiorbdModel

try:
    from .osim_model_interface import OsimModelNoMesh, OsimModel
except ImportError:
    # OpenSim n'est pas installé, ces classes ne seront pas disponibles
    pass

from .available_interfaces import model_from_file
