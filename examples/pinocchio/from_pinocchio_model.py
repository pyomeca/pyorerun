"""
Example of using Pinocchio model with pyorerun.

This example demonstrates how to load and animate a URDF model (Baxter robot) using the Pinocchio interface.
"""

import numpy as np
from pathlib import Path

# Check if pinocchio is available
try:
    import pinocchio as pin
    from pyorerun import PhaseRerun, PinocchioModel
except ImportError as e:
    print(f"Error: {e}")
    print("Please install pinocchio: pip install pin")
    exit(1)


def main():
    # Use the Baxter robot model from the examples
    # Get the path relative to this script
    script_dir = Path(__file__).parent
    model_path = script_dir / "urdf" / "baxter_local.urdf"

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    # Load the Pinocchio model
    print(f"Loading model from: {model_path}")
    pinocchio_model = PinocchioModel(str(model_path))

    # Create a simple animation
    nb_frames = 200
    nb_seconds = 2
    t_span = np.linspace(0, nb_seconds, nb_frames)

    # Generate some generalized coordinates
    # For a standing position, we need to generate appropriate q values
    q = np.zeros((pinocchio_model.nb_q, nb_frames))

    # Add some simple motion (sinusoidal movement on some joints)
    for i in range(min(3, pinocchio_model.nb_q)):
        q[i, :] = 0.1 * np.sin(2 * np.pi * t_span * (i + 1))

    # Create the visualization
    print(f"Model name: {pinocchio_model.name}")
    print(f"Number of DoFs: {pinocchio_model.nb_q}")
    print(f"Number of segments: {pinocchio_model.nb_segments}")
    print(f"Number of markers: {pinocchio_model.nb_markers}")

    viz = PhaseRerun(t_span)
    viz.add_animated_model(pinocchio_model, q)
    viz.rerun("pinocchio_model_example")


if __name__ == "__main__":
    main()
