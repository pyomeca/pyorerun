"""
This example is a simple example of how to use the LiveModelAnimation class to animate a model in real-time.
The user can interact with the model by changing the joint angles using sliders.
"""

import rerun

from pyorerun import LiveModelAnimation
import numpy as np


model_path = "D:\Documents\Programmation\osim_to_biomod\example\Models\Model_Pose2Sim.osim"
import rerun as rr
import numpy as np
from pyorerun import ModelUpdater

model = ModelUpdater.from_file(model_path)
rr.init("my_thing", spawn=True)
# rr.log("test", timeless=True)
q = np.zeros(model.model.nb_q)
for i in range(10):
    rr.set_time_sequence(timeline="step", sequence=i)
    model.to_rerun(q)
    rr.log("points", rr.Points3D(np.random.rand(10, 3)))
# add random 3D points

# rr.log("anything", rr.Anything())
