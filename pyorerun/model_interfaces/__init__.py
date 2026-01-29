from .abstract_model_interface import AbstractSegment, AbstractModel, AbstractModelNoMesh

# Biorbd
try:
    from .biorbd_model_interface import BiorbdModelNoMesh, BiorbdModel
except ImportError:
    # biorbd is not installed, these classes will not be available
    pass


# Opensim
try:
    from .osim_model_interface import OsimModelNoMesh, OsimModel
except ImportError:
    # OpenSim is not installed, these classes will not be available
    pass

# Biobuddy
try:
    from .biobuddy_model_interface import BiobuddyModelNoMesh, BiobuddyModel
except ImportError:
    # BioBuddy is not installed, these classes will not be available
    pass

# Pinocchio
try:
    from .pinocchio_model_interface import PinocchioModelNoMesh, PinocchioModel
except ImportError:
    # Pinocchio is not installed, these classes will not be available
    pass

from .available_interfaces import model_from_file
