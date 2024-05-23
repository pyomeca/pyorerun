import pyorerun as prr

prr.c3d(
    "Running_0002.c3d",
    show_floor=True,
    show_force_plates=True,
    show_forces=True,
    down_sampled_forces=True,
    video=("Running_0002_Oqus_6_15004.avi", "Running_0002_Oqus_9_15003.avi"),
)
