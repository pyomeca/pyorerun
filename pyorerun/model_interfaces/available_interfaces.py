from .abstract_model_interface import AbstractModel, AbstractModelNoMesh
from .biorbd_model_interface import BiorbdModelNoMesh, BiorbdModel
from ..model_components.model_display_options import DisplayModelOptions

AVAILABLE_INTERFACES = {
    "biorbd": (BiorbdModel, BiorbdModelNoMesh),
}

try:
    from .osim_model_interface import OsimModelNoMesh, OsimModel

    AVAILABLE_INTERFACES["opensim"] = (OsimModel, OsimModelNoMesh)
except ImportError:
    # OpenSim n'est pas installé, ces classes ne seront pas disponibles
    pass

try:
    from .pinocchio_model_interface import PinocchioModelNoMesh, PinocchioModel

    AVAILABLE_INTERFACES["pinocchio"] = (PinocchioModel, PinocchioModelNoMesh)
except ImportError:
    # Pinocchio is not installed, these classes will not be available
    pass


def model_from_file(model_path: str, options: DisplayModelOptions = None) -> tuple[AbstractModel, AbstractModelNoMesh]:
    """
    Create a model interface based on the provided model path.

    Parameters
    ----------
    model_path : str
        Path to the model file (either .osim or .bioMod).
    options : dict, optional
        Options for the model interface.

    Returns
    -------
    tuple[AbstractModel, AbstractModelNoMesh]
        A tuple containing the model interface and a no-mesh instance of the model.
    """

    if model_path.endswith(".osim"):
        if "opensim" not in AVAILABLE_INTERFACES:
            raise ImportError(
                f"OpenSim is not installed. Please install it to use OpenSim models."
                f"Use: conda install opensim-org::opensim"
            )
        model = AVAILABLE_INTERFACES["opensim"][0](model_path, options=options)
        no_instance_mesh = AVAILABLE_INTERFACES["opensim"][1]
    elif model_path.endswith(".bioMod"):
        model = AVAILABLE_INTERFACES["biorbd"][0](model_path, options=options)
        no_instance_mesh = AVAILABLE_INTERFACES["biorbd"][1]
    elif model_path.endswith(".urdf"):
        if "pinocchio" not in AVAILABLE_INTERFACES:
            raise ImportError(
                f"Pinocchio is not installed. Please install it to use URDF models."
                f"Use: pip install pin (or conda install pinocchio)"
            )
        model = AVAILABLE_INTERFACES["pinocchio"][0](model_path, options=options)
        no_instance_mesh = AVAILABLE_INTERFACES["pinocchio"][1]
    else:
        raise ValueError("The model must be in biorbd (.bioMod), opensim (.osim), or pinocchio (.urdf) format.")

    return model, no_instance_mesh
