"""
Move the jao2 Mujoco arm to a target position.
The simulation ends after 1500 time steps, and the
trajectory of the end-effector is plotted in 3D.
"""
import sys
import traceback

import glfw
import numpy as np

from abr_control.controllers import Damping
from my_osc import OSC
from mujoco_interface import Mujoco
from abr_control.utils import transformations
from my_mujoco_config import MujocoConfig as arm

robot_config = arm('ur5.xml', folder='./my_models/ur5_robotiq85')

# create our Mujoco interface
interface = Mujoco(robot_config, dt=0.02)
interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)
print(type(robot_config.START_ANGLES))

interface.send_target_angles(robot_config.START_ANGLES)

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# instantiate controller
ctrlr = OSC(
    robot_config,
    kp=200,
    null_controllers=[damping],
    vmax=[0.5, 0.5],  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, True, True, True],
)

# set up lists for tracking data
ee_track = []
target_track = []

target_geom_id = interface.sim.model.geom_name2id("target2")
erw = [0, 0.9, 0, 1]
red = [0.9, 0, 0, 1]


def gen_target(interface):
    # target_xyz = (np.random.rand(2) +  + np.array([-0.5, -0.5, 0.5])) * np.array(
    #     [1, 1, 0.5]
    # )
    target_offset_xy = np.random.rand(2) * 0.5
    print(target_offset_xy)
    interface.sim.data.qpos[-7] = (np.random.rand(1) - 0.5) * 1
    interface.sim.data.qpos[-6] = (np.random.rand(1) - 0.5) * 0.25 + 0.7
    # interface.sim.data.qpos[-7] = 0.75
    # interface.sim.data.qpos[-6] = 0
    
gen_target(interface)

# observation = (joint0, joint1, joint2, joint3, joint4, joint5, 
#                finger_joint, finger_joint2, finger_joint3, finger_joint4, finger_joint5, finger_joint6,
#                target_x, target_y, target_x, target_r, target_x, target_y, target_z)
observation = interface.sim.data.qpos
print(observation)






try:
    # get the end-effector's initial position
    feedback = interface.get_feedback()
    start = robot_config.Tx("EE", feedback["q"])

    # make the target offset from that start position
    # gen_target(interface)

    count = 0.0
    print("\nSimulation starting...\n")

    #######################################################################
    # Go to the object
    #######################################################################
    while 1:
        if interface.viewer.exit:
            glfw.destroy_window(interface.viewer.window)
            break
        # get joint angle and velocity feedback
        feedback = interface.get_feedback()

        # target = np.hstack(
        #     [
        #         interface.get_xyz("target2"),
        #         transformations.euler_from_quaternion(
        #             interface.get_orientation("target2"), "rxyz"
        #         ),
        #     ]
        # )

        print("eef orientation", transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz"))


        target_xyz = interface.get_xyz("target2")
        target_xyz[-2] -= 0.15

        # target_xyz[-1] = 0.5
        # target_xyz[-2] = 0.5
        # target_xyz[-3] = 0.5
        target = np.hstack(
            [
                target_xyz,
                # (0.5, 0.4, 0.05),
                # (0, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )
        print(transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz"))
        print(transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz"))

        # calculate the control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )

        u[-1] = -0.05

        # send forces into Mujoco, step the sim forward
        if True:
            interface.send_forces(u)

        # calculate end-effector position
        ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(target[:3]))

        error = np.linalg.norm(ee_xyz - target[:3])
        if error < 0.02:
            # interface.sim.model.geom_rgba[target_geom_id] = erw
            # count += 1
            break
            # continue
        # else:
        #     count = 0
        #     # interface.sim.model.geom_rgba[target_geom_id] = red

        # if count >= 50:
        #     # gen_target(interface)
        #     count = 0
        count += 1


    #######################################################################
    # Put the object in position
    #######################################################################
    while 1:
        if interface.viewer.exit:
            glfw.destroy_window(interface.viewer.window)
            break
        # get joint angle and velocity feedback
        feedback = interface.get_feedback()

        print("eef orientation", transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz"))

        target_xyz = interface.get_xyz("target2")
        # target_xyz[-2] -= 0.02
        target = np.hstack(
            [
                target_xyz,
                # (0.5, 0.4, 0.05),
                # (0, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
                # (3.14, 0, 1.57),
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )

        # calculate the control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )
        u[-1] = -0.05

        # send forces into Mujoco, step the sim forward
        interface.send_forces(u)

        # calculate end-effector position
        ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(target[:3]))

        error = np.linalg.norm(ee_xyz - target[:3])
        if error < 0.005:
            # interface.sim.model.geom_rgba[target_geom_id] = green
            count += 1
            break
        else:
            count = 0
            # interface.sim.model.geom_rgba[target_geom_id] = red

        if count >= 50:
            # gen_target(interface)
            count = 0


    #######################################################################
    # Grip the object
    #######################################################################
    while 1:
        if interface.viewer.exit:
            glfw.destroy_window(interface.viewer.window)
            break
        # get joint angle and velocity feedback
        feedback = interface.get_feedback()

        print("eef orientation", transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz"))


        target_xyz = interface.get_xyz("target2")
        # target_xyz[-2] -= 0.02
        target = np.hstack(
            [
                target_xyz,
                # (0.5, 0.4, 0.05),
                # (0, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
                # (3.14, 0, 1.57),

                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )

        # calculate the control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )

        u[-1] = 0.1
        # send forces into Mujoco, step the sim forward
        interface.send_forces(u)

        # calculate end-effector position
        ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(target[:3]))

        error = np.linalg.norm(ee_xyz - target[:3])
        # if error < 0.02:
        #     interface.sim.model.geom_rgba[target_geom_id] = green
        #     count += 1
        # else:
        #     count = 0
        #     interface.sim.model.geom_rgba[target_geom_id] = red

        count += 1
        if count >= 100:
            # gen_target(interface)
            count = 0
            break


    #######################################################################
    # Bring the object up
    #######################################################################
    while 1:
        if interface.viewer.exit:
            glfw.destroy_window(interface.viewer.window)
            break
        # get joint angle and velocity feedback
        feedback = interface.get_feedback()

        print("eef orientation", transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz"))

        target_xyz = interface.get_xyz("target2")
        target_xyz[-1] = 0.3
        # target_xyz[-2] -= 0.02
        target = np.hstack(
            [
                target_xyz,
                # (0.5, 0.4, 0.05),
                # (0, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
                # (3.14, 0, 1.57),

                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )


        # calculate the control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )

        u[-1] = 0.1
        # send forces into Mujoco, step the sim forward
        interface.send_forces(u)

        # calculate end-effector position
        ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(target[:3]))

        error = np.linalg.norm(ee_xyz - target[:3])
        if error < 0.02:
            # interface.sim.model.geom_rgba[target_geom_id] = green
            count += 1
        else:
            count = 0
            # interface.sim.model.geom_rgba[target_geom_id] = red

        if count >= 50:
            # gen_target(interface)
            count = 0

except:
    print(traceback.format_exc())
