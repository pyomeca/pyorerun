# Pyorerun
[![Maintainability](https://api.codeclimate.com/v1/badges/7e8b7eb962759cf11f38/maintainability)](https://codeclimate.com/github/pyomeca/pyorerun/maintainability) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyorerun.svg)](https://anaconda.org/conda-forge/pyorerun)

We can rerun c3d files and display their original content.
And also animate biorbd models from the pyomeca organization.

``` conda install -c conda-forge pyorerun rerun-sdk=0.21```

# Rerun .c3d - As simple as that

``` python3
import pyorerun as prr

prr.c3d("example.c3d")
```

https://github.com/pyomeca/pyorerun/assets/40755537/3cdd39f6-d88d-4891-8ffd-d14c93a9c94e

**NOTE**: Only handle markers, force plates, floor for now

## Notebook demo


https://github.com/pyomeca/pyorerun/assets/40755537/c9448d69-c891-4362-821b-68449684eb64



# Rerun Biorbd Models

``` python3
from pyorerun import BiorbdModel, PhaseRerun

nb_frames = 10
nb_seconds = 0.1
t_span = np.linspace(0, nb_seconds, nb_frames)

model = BiorbdModel("models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod")
q = np.zeros((model.model.nbQ(), nb_frames))

viz = PhaseRerun(t_span)
viz.add_animated_model(model, q)
viz.rerun("msk_model")
```


https://github.com/pyomeca/pyorerun/assets/40755537/9341707a-4f44-4dbd-8ece-9a16e1500e9a

# Play with joint DoFs q, Live.
``` python3
from pyorerun import LiveModelAnimation


model_path = "models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod"
animation = LiveModelAnimation(model_path, with_q_charts=True)
animation.rerun()
```

## From source
```conda install -c conda-forge ezc3d rerun-sdk=0.21 trimesh numpy biorbd pyomeca tk imageio imageio-ffmpeg```
```conda install -c opensim-org::opensim```

Then, ensure it is accessible in your Python environment by installing the package:

``` pip install . ``` or ``` python setup.py install ```

## On Linux

Rerun should work out-of-the-box on Mac and Windows, but on Linux, you need first to run:

```
sudo apt-get -y install \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    patchelf
```

# Citing
```
@software{pierre_puchaud_2024_11449215,
  author       = {Pierre Puchaud, Mickael Begon},
  title        = {pyomeca/pyorerun: CaffeineStrips},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.2.4},
  doi          = {10.5281/zenodo.11449215},
  url          = {https://doi.org/10.5281/zenodo.11449215}
}
```

# Contributing
Contributions are welcome. I will be happy to review and help you to improve the code.

