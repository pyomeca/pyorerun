# Rerun C3D
Rerun C3D is a tool to rerun the C3D file and display its original content.

# Installation prerequisites
``` conda install -c conda-forge ezc3d rerun-sdk ```

Then, download the rrc3d script from the provided location and ensure it is accessible in your Python environment.

# As simple as that

``` python3
from rrc3d import rrc3d

rrc3d("example.c3d")
```

<p align="center">
    <img
      src="docs/rerun-c3d-viewer.png"
      alt="logo"
      width="500"
    />
</p>

# NOTE
- Only handle markers for now

# Contributing
Contributions to the rrc3d script are welcome.
Please follow the standard GitHub pull request process to submit your contributions.
