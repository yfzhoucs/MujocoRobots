import glfw
import numpy as np


class GripperStatus:
    def __init__(self, gripper_status):
        self.gripper_status = gripper_status

    def get_gripper_status(self):
        return self.gripper_status

    def set_gripper_status(self, new_gripper_status):
        self.gripper_status = new_gripper_status


class Action:
    def __init__(self, interface, controller):
        self.interface = interface
        self.controller = controller
        return
    
    # abstract method in base class
    # https://stackoverflow.com/questions/4382945/abstract-methods-in-python
    def execute(self):
        raise NotImplementedError("Please Implemente this Method")


class MoveTo(Action):
    def __init__(self, interface, controller, target_func, gripper_control_func, time_limit=None, error_limit=None):
        assert time_limit is not None or error_limit is not None, "at least 1 of time limit or error limit should be indicated"

        super().__init__(interface, controller)
        self.target_func = target_func
        self.gripper_control_func = gripper_control_func
        self._gripper = None
        self.time_limit = time_limit
        self.error_limit = error_limit

    def _set_gripper(self, gripper):
        self._gripper = gripper

    def execute(self):
        time_step = 0
        while True:

            # Check Window
            if self.interface.viewer.exit:
                glfw.destroy_window(self.interface.viewer.window)
                break

            # Get Target
            target = self.target_func(self.interface)

            # Calculate Forces
            feedback = self.interface.get_feedback()
            u = self.controller.generate(
                q=feedback["q"],
                dq=feedback["dq"],
                target=target,
            )

            # Set gripper force
            if self._gripper is not None:
                u = self.gripper_control_func(u, self._gripper)

            # send forces into Mujoco, step the sim forward
            self.interface.send_forces(u)

            # calculate time step
            time_step += 1
            # calculate error
            # ee_xyz = robot_config.Tx("EE", q=feedback["q"])
            ee_xyz = self.interface.get_xyz("EE")
            error = np.linalg.norm(ee_xyz - target[:3])

            # whether stop criterion has been reached
            if self.time_limit is not None:
                if time_step >= self.time_limit:
                    break
            if self.error_limit is not None:
                if error <= self.error_limit:
                    break
        

class Executor:
    def __init__(self, interface, start_angles, start_gripper_status):
        self.interface = interface
        self.action_list = []
        interface.send_target_angles(start_angles)
        self.gripper = GripperStatus(start_gripper_status)

    def append(self, action):
        action._set_gripper(self.gripper)
        self.action_list.append(action)

    def execute(self):
        for i in range(len(self.action_list)):
            self.action_list[i].execute()

    def init_env(self):
        self.stage = 0
        return state, img

    def step(self, action):
        return state, end

    def plan_action(self, state):
        return action, end


if __name__ == '__main__':

    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm

    def target_func(interface):
        target_xyz = interface.get_xyz("target2")
        target_xyz[-2] -= 0.15
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def target_func_2(interface):
        target_xyz = interface.get_xyz("target2")
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def target_func_3(interface):
        target_xyz = interface.get_xyz("EE")
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def target_func_4(interface):
        target_xyz = interface.get_xyz("EE")
        target_xyz[-1] = 0.3
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation("target2"), "rxyz")[-1] + 1.57),
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

    def gripper_control_func_3(u, gripper):
        gripper.set_gripper_status(-0.05)
        u[-1] = gripper.get_gripper_status()
        return u

    # create our Mujoco interface
    robot_config = arm('ur5.xml', folder='./my_models/ur5_robotiq85')
    interface = Mujoco(robot_config, dt=0.008)
    interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=-1)

    # Randomly place the target object
    def gen_target(interface):
        target_offset_xy = np.random.rand(2) * 0.5
        interface.sim.data.qpos[-7] = (np.random.rand(1) - 0.5) * 0.7
        interface.sim.data.qpos[-6] = (np.random.rand(1) - 0.5) * 0.25 + 0.7475 - 0.3
    # def gen_target(interface):
    #     interface.sim.data.qpos[-7] = 0.35
    #     interface.sim.data.qpos[-6] = (np.random.rand(1) - 0.5) * 0 + 0.75 - 0.3 - 0.15 + 0.25
    gen_target(interface)
    
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
    e.append(MoveTo(interface, ctrlr, target_func_3, gripper_control_func_3, time_limit=25))
    e.execute()

    e.execute()

    end = False
    state, img = e.init_env()
    while not end:
        action, end = e.plan_action(state)
        state, img = e.step(action)


