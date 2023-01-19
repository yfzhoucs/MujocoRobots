#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, load_model_from_path, MjSim, MjViewer
import mujoco_py
import math
import os
import matplotlib.pyplot as plt

# model = load_model_from_path('./my_models/ur5_abr/ur5.xml')
model = load_model_from_path('./my_models/ur5_robotiq85/ur5.xml')
sim = MjSim(model)
viewer1 = MjViewer(sim)



# print(sim.viewer)
viewer = mujoco_py.MjRenderContextOffscreen(sim)
# viewer.cam.lookat[1] = 1
# viewer.cam.lookat[2] = 1

viewer.cam.fixedcamid = model.camera_name2id('111')
viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED

# print(viewer.cam.lookat)
# print(sim.data.ctrl.shape)
# print(sim.data.sensordata)
# exit()

t = 0
while True:
    print(sim.data.ctrl.shape)
    sim.data.ctrl[0] = 0.5
    sim.data.ctrl[1] = 0.5
    sim.data.ctrl[2] = 0.5
    sim.data.ctrl[3] = 0.5
    sim.data.ctrl[4] = 0.5
    sim.data.ctrl[5] = 0.5
    sim.data.ctrl[5] = 0.5
    t += 1
    sim.step()
    # print(sim.get_state())
    # viewer.render()
    viewer.render(1920, 1080, 0)
    rgb = viewer.read_pixels(1920, 1080)[0]
    print(rgb)
    plt.imshow(rgb)
    plt.show()
    # exit()


    # print(a)
    # img = sim.render(224, 224, camera_name='111')
    # print(img)
    # plt.imshow(img)
    # plt.show()
    # print()
    # print(sim.data.sensordata)
