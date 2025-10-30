from setuptools import setup

setup(
    name="pyorerun",
    version="1.5.0",
    install_requires=[
        "ezc3d",  # Not yet available on pypi, use `conda install -c conda-forge ezc3d`
        "numpy",
        "rerun-sdk=0.24.1",
        "trimesh",
        "tk",
        "imageio",
        "imageio-ffmpeg",
        "biorbd",  # Not yet available on pypi, use `conda install -c conda-forge biorbd`
    ],
    author="Pierre Puchaud",
    author_email="puchaud.pierre@gmail.com",
    maintainer="Pierre Puchaud",
    maintainer_email="puchaud.pierre@gmail.com",
    description="A Python package to rerun C3D files and biomechanical simulations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="LICENSE.md",
    keywords=["c3d", "motion capture", "rerun", "biorbd", "markers"],
    url="http://github.com/Ipuch/pyorerun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
