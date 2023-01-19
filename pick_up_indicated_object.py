from sequential_actions_interface import *


if __name__ == '__main__':

    import numpy as np
    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm

    targets = ["target1", "target2", "target3"]
    target_name = targets[np.random.randint(len(targets))]

    def target_func(interface):
        target_xyz = interface.get_xyz(target_name)
        target_xyz[-2] -= 0.15
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation(target_name), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def target_func_2(interface):
        target_xyz = interface.get_xyz(target_name)
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation(target_name), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def target_func_3(interface):
        target_xyz = interface.get_xyz("EE")
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation(target_name), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def target_func_4(interface):
        target_xyz = interface.get_xyz("EE")
        target_xyz[-1] = 0.3
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation(target_name), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def gripper_control_func(u, gripper):
        u[-1] = gripper.get_gripper_status()
        return u

    def gripper_control_func_2(u, gripper):
        gripper.set_gripper_status(0.1)
        u[-1] = gripper.get_gripper_status()
        return u

    # create our Mujoco interface
    robot_config = arm('ur5_3_objects.xml', folder='./my_models/ur5_robotiq85')
    interface = Mujoco(robot_config, dt=0.008)
    interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)

    # Randomly place the target object
    def gen_target(interface):

        def l2(pos1, pos2):
            assert len(pos1) == len(pos2)
            d = 0
            for i in range(len(pos1)):
                d = d + (pos1[i] - pos2[i]) ** 2
            d = d ** (1 / 2)
            return d

        point_list = []

        for i in range(3):
            while True:
                x = (np.random.rand(1) - 0.5) * 0.7
                y = (np.random.rand(1) - 0.5) * 0.25 + 0.7475 - 0.3
                too_close = False
                for j in range(len(point_list)):
                    if l2(point_list[j], (x, y)) < 0.1 or abs(point_list[j][0] - x) < 0.1:
                        too_close = True
                if not too_close:
                    point_list.append((x, y))
                    interface.sim.data.qpos[-7 - i * 7] = x
                    interface.sim.data.qpos[-6 - i * 7] = y
                    break

    gen_target(interface)
    print(interface.sim.data.qpos)
    # exit()
    
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

    e = Executor(interface, robot_config.START_ANGLES, -0.05)
    e.append(MoveTo(interface, ctrlr, target_func, gripper_control_func, error_limit=0.02))
    e.append(MoveTo(interface, ctrlr, target_func_2, gripper_control_func, error_limit=0.02))
    e.append(MoveTo(interface, ctrlr, target_func_3, gripper_control_func_2, time_limit=25))
    e.append(MoveTo(interface, ctrlr, target_func_4, gripper_control_func, error_limit=0.02))
    e.execute()
