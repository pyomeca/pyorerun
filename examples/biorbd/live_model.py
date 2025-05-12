"""
This example is a simple example of how to use the LiveModelAnimation class to animate a model in real-time.
The user can interact with the model by changing the joint angles using sliders.
"""

import rerun

from pyorerun import LiveModelAnimation
import numpy as np


model_path = "models/Wu_Shoulder_Model_kinova_scaled_adjusted_2.bioMod"
import rerun as rr
import numpy as np
from pyorerun import ModelUpdater

model = ModelUpdater.from_file(model_path)
rr.init("my_thing", spawn=True)
# rr.log("test", timeless=True)
q = np.zeros(16)
for i in range(100):
    rr.set_time_sequence(timeline="step", sequence=i)
    model.to_rerun(q)
# rr.log("anything", rr.Anything())
