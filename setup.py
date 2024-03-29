from setuptools import setup

setup(
    name="pyorerun",
    version="0.2.1",
    # packages=find_packages(),
    install_requires=[
        "ezc3d",
        "numpy",
        "rerun-sdk",
        "trimesh",
        "biorbd",
        "pyomeca",
    ],
    author="Pierre Puchaud",
    author_email="puchaud.pierre@gmail.com",
    maintainer="Pierre Puchaud",
    maintainer_email="puchaud.pierre@gmail.com",
    description="A Python package to rerun C3D files and pyomeca simulations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="LICENSE",
    keywords=["c3d", "motion capture", "rerun", "biorbd", "pyomeca"],
    url="http://github.com/Ipuch/pyorerun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
