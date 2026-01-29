# OpenSim Examples

These examples demonstrate how to use `pyorerun` with OpenSim models.

## Installation

Using `pyorerun` with OpenSim requires a careful installation procedure due to multiple dependency conflicts.

### Why the special setup?

There are **two conflicts** to navigate:

1. **ezc3d conflict**: Both `pyorerun` and `opensim` need `ezc3d`, but in different ways:
   - `pyorerun` needs Python bindings (`import ezc3d`)
   - `opensim` needs the C++ library (`libezc3d.so`)
   
   The conda ezc3d package has broken Python bindings, so we overlay pip's version.

2. **casadi conflict**: The `pyorerun` conda package pulls in `casadi` through its dependencies, which breaks OpenSim's Moco module (`libosimMoco.so`).

### Step-by-step installation

To avoid conflicts, install opensim first, then pyorerun **from source**:

```bash
# 1. Create a fresh conda environment with Python
conda create -n pyorerun-osim python=3.12
conda activate pyorerun-osim

# 2. Install opensim and ezc3d FIRST (before pyorerun)
conda install -c opensim-org -c conda-forge opensim ezc3d

# 3. Overlay pip's ezc3d Python bindings (fixes the broken conda bindings)
pip install ezc3d --force-reinstall --no-deps

# 4. Install pyorerun from source (avoids conda's casadi conflict)
git clone https://github.com/Ipuch/pyorerun.git
pip install ./pyorerun

# 5. Verify everything works
python -c "import ezc3d; import opensim; import pyorerun; print('All imports OK')"
```

> [!IMPORTANT]
> - The `--no-deps` flag is critical for step 3 — it installs only ezc3d without changing numpy.
> - Installing `pyorerun` from source (step 4) avoids conda pulling in conflicting casadi.

### If you don't need OpenSim's Moco module

If you only need basic OpenSim functionality (not Moco), you can use a simpler install:

```bash
conda create -n pyorerun-osim python=3.12
conda activate pyorerun-osim
conda install -c conda-forge -c opensim-org pyorerun opensim ezc3d
pip install ezc3d --force-reinstall --no-deps
```

Then import opensim modules selectively (avoid `from opensim import *`):
```python
import opensim
model = opensim.Model("model.osim")  # This works
# But avoid: opensim.MocoStudy()  # This may fail due to casadi conflict
```

## Running the examples

```bash
cd examples/osim
python from_osim_model.py
```
