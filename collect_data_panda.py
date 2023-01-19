import glfw
import numpy as np
import json
import numpy as np
from abr_control.controllers import Damping
from abr_control.controllers import Joint
from my_osc import OSC
from mujoco_interface import Mujoco
from abr_control.utils import transformations
from my_mujoco_config import MujocoConfig as arm
import os
import matplotlib.pyplot as plt
from mujoco_py import MjRenderContextOffscreen
import cv2


global_target_xyz = None
target_func_6_init = False
target_func_4_init = False
target_func_above_target_pick_up_init = False
target_func_place_init = False
target_func_turn_over_init = False
target_func_release_init = False
RESOLUTION = 224


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
            if hasattr(self.interface, 'viewer'):
                if self.interface.viewer.exit:
                    glfw.destroy_window(self.interface.viewer.window)
                    return

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

    def ends(self, state):
        # calculate error
        # ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        ee_xyz = self.interface.get_xyz("EE")
        # print('ee_xyz', ee_xyz)
        # print('target', state['target'])
        error = np.linalg.norm(ee_xyz - state['target'][:3])

        end = False
        # whether stop criterion has been reached
        if self.time_limit is not None:
            if state['step_in_stage'] >= self.time_limit:
                end = True
        if self.error_limit is not None:
            if error <= self.error_limit:
                end = True
        return end

    def plan_action(self, state):
        action = self.controller.generate(
            q=state["q"],
            dq=state["dq"],
            target=state['target'],
        )


        # Set gripper force
        if self._gripper is not None:
            action = self.gripper_control_func(action, self._gripper)

        end = self.ends(state)
        return action, end
        

class Executor:
    def __init__(self, interface, start_angles, start_gripper_status, object_keyword='target'):
        self.interface = interface
        self.action_list = []
        interface.send_target_angles(start_angles)
        self.gripper = GripperStatus(start_gripper_status)
        self.object_keyword = object_keyword

    def append(self, action):
        action._set_gripper(self.gripper)
        self.action_list.append(action)

    def execute(self):
        for i in range(len(self.action_list)):
            self.action_list[i].execute()

    def get_objects_to_track(self):
        objects_to_track = {}
        for object_name in self.objects_to_track:
            objects_to_track[object_name] = {
                'xyz': self.interface.get_xyz(object_name),
                'rpy': transformations.euler_from_quaternion(self.interface.get_orientation(object_name), "rxyz")
            }
        return objects_to_track

    def get_state(self):
        # Check Window
        if hasattr(self.interface, 'viewer'):
            if self.interface.viewer.exit:
                glfw.destroy_window(self.interface.viewer.window)
                return

        state = {}
        feedback = self.interface.get_feedback()
        state['q'] = feedback['q']
        state['dq'] = feedback['dq']
        state['target'] = self.action_list[self.stage].target_func(self.interface)
        state['objects_to_track'] = self.get_objects_to_track()
        state['stage'] = self.stage
        state['step_in_stage'] = self.step_in_stage
        state['goal_object'] = self.goal_object
        state['action_inst'] = self.action_inst
        return state

    def render_img(self):
        # img = self.interface.sim.render(224, 224, camera_name='111')

        # img = self.interface.viewer.read_pixels(224, 224)[0]

        self.offscreen.render(RESOLUTION, RESOLUTION, 0)
        rgbd_image = self.offscreen.read_pixels(RESOLUTION, RESOLUTION)
        rgbd_image = (rgbd_image[0], self._convert_depth_to_meters(self.interface.sim, rgbd_image[1]))
        return rgbd_image

    def init_env(self, objects_to_track, goal_object, action_inst):
        self.stage = 0
        self.step_in_stage = 0
        self.objects_to_track = objects_to_track
        self.goal_object = goal_object
        self.action_inst = action_inst
        state = self.get_state()
        img = self.render_img()
        # img = None
        return state, img

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def _convert_depth_to_meters(self, sim, depth):
        extent = sim.model.stat.extent
        near = sim.model.vis.map.znear * extent
        far = sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def step(self, action):
        # Check Window
        if hasattr(self.interface, 'viewer'):
            if self.interface.viewer.exit:
                glfw.destroy_window(self.interface.viewer.window)
                return

        # Execute the action
        self.interface.send_forces(action)
        self.step_in_stage += 1

        # Get state and image
        state = self.get_state()
        img = self.render_img()

        return state, img

    def plan_action(self, state):
        action, end = self.action_list[self.stage].plan_action(state)
        print(action, self.stage, self.action_inst)

        # Keep track of stages/phases
        if end:
            self.stage += 1
            self.step_in_stage = 0
            print('stage', self.stage)

        # See whether all the actions are done
        if self.stage == len(self.action_list):
            end = True
        else:
            end = False

        return action, end


class Recorder:
    def __init__(self, data_dir, data_id):
        self.data_dir = data_dir
        self.data_id = data_id
        self.state_seq = []
        self.id_dir = os.path.join(data_dir, str(data_id))
        if not os.path.isdir(self.id_dir):
            os.mkdir(self.id_dir)

    def record_state(self, state):
        self.state_seq.append(state)

    def save_states(self):    
        with open(self.id_dir + '/states.json', 'w') as f:
            json.dump(self.state_seq, f, cls=NumpyEncoder, indent=4)

    def save_img(self, img, step, scale=1000):
        # plt.imsave(self.id_dir + '/' + str(step) + '.png', img)
        rgb_img = img[0]
        d_img = img[1]
        rgb_img = rgb_img[::-1, :, :]
        d_img = d_img[::-1, :]
        d_img = np.uint16(d_img * scale)
        plt.imsave(self.id_dir + '/' + str(step) + '.png', rgb_img)
        np.save(self.id_dir + '/' + str(step) + '_depth_map.npy', d_img)

    def record_step(self, img, state, step):
        self.save_img(img, step)
        # input()
        self.record_state(state)


# Serialize numpy arrays
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def init_env(robot_config, interface):

    targets = ['target2', 'coke', 'pepsi', 'milk', 'bread', 'bottle']
    target_name = targets[np.random.randint(len(targets))]

    def target_func(interface):
        target_xyz = interface.get_xyz(target_name)
        target_xyz[-1] += 0.20
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target

    def target_func_2(interface):
        target_xyz = interface.get_xyz(target_name)
        target_xyz[-2] += 0.03
        target_xyz[-1] += 0.01
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, 1.57),
            ]
        )
        return target

    def target_func_3(interface):
        target_xyz = interface.get_xyz("EE")
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, 1.57),
            ]
        )
        return target

    def target_func_4(interface):
        global global_target_xyz
        global target_func_4_init
        if not target_func_4_init:
            target_xyz = interface.get_xyz("EE")
            target_xyz[-1] = 0.2
            global_target_xyz = target_xyz
            target_func_4_init = True
        target = np.hstack(
            [
                global_target_xyz,
                (3.14, 0, 1.57),
            ]
        )
        return target

    def target_func_5(interface):
        target_xyz = interface.get_xyz("EE")
        target = np.hstack(
            [
                target_xyz + np.random.rand(3) * np.array([5, 10, 10]) - np.array([0, 5, 5]),
                (3.14, 0, 1.57),
            ]
        )
        return target

    def target_func_6(interface):
        global global_target_xyz
        global target_func_6_init
        if not target_func_6_init:
            target_xyz = interface.get_xyz(target_name)
            target_xyz[-1] = 0.07
            target_xyz[-2] += 0.15
            global_target_xyz = target_xyz
            target_func_6_init = True
        target = np.hstack(
            [
                global_target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target

    def target_func_above_target_approach(interface):
        target_xyz = interface.get_xyz(target_name)
        # if target_xyz[-3] < 0:
        #     angle = (3.14, 0, 0)
        #     target_xyz[-3] += 0.15
        # else:
        #     angle = (3.14, 0, -3.14)
        #     target_xyz[-3] -= 0.15

        target_xyz[-1] += 0.15
        angle = (1.57, 3.14, 1.57)

        target = np.hstack(
            [
                target_xyz,
                angle
            ]
        )
        return target

    def target_func_joint_from_above(interface):
        return (-0.0628, 0.405, -1.04, 0.0628, -1.57, 0, 0.8)

    def target_func_joint_from_above_init(interface):
        return (-0.0628, -0.565, -2.42, -0.314, -1.57, 0, 0.8)

    def target_func_above_target_move_in(interface):
        target_xyz = interface.get_xyz(target_name)

        target_xyz[-1] += 0.03
        angle = (3.14, 0, -1.57)

        target = np.hstack(
            [
                target_xyz,
                angle
            ]
        )
        return target

    def target_func_above_target_move_in_bottle(interface):
        target_xyz = interface.get_xyz(target_name)

        target_xyz[-1] += 0.07
        angle = (3.14, 0, -1.57)

        target = np.hstack(
            [
                target_xyz,
                angle
            ]
        )
        return target

    def target_func_above_target_move_in_milk(interface):
        target_xyz = interface.get_xyz(target_name)

        target_xyz[-1] += 0.06
        angle = (3.14, 0, -1.57)

        target = np.hstack(
            [
                target_xyz,
                angle
            ]
        )
        return target

    def target_func_above_target_pick_up(interface):
        global global_target_xyz
        global target_func_above_target_pick_up_init
        if not target_func_above_target_pick_up_init:
            target_xyz = interface.get_xyz("EE")
            target_xyz[-1] = 0.2
            global_target_xyz = target_xyz
            target_func_4_init = True
        target = np.hstack(
            [
                global_target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target


    def target_func_move_behind(interface):
        target_xyz = interface.get_xyz(target_name)
        target_xyz[-2] -= 0.15
        target_xyz[-1] = 0.07
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target


    def target_func_turn_over(interface):
        global global_target_xyz
        global target_func_turn_over_init
        if not target_func_turn_over_init:
            target_xyz = interface.get_xyz("EE")
            target_xyz[-2] -= 0.15
            global_target_xyz = target_xyz
            target_func_turn_over_init = True
        target = np.hstack(
            [
                global_target_xyz,
                (1.57, 0, -1.57),
            ]
        )
        return target

    def target_func_place(interface):
        global global_target_xyz
        global target_func_place_init
        if not target_func_place_init:
            target_xyz = interface.get_xyz("EE")
            target_xyz[-1] = 0.05
            global_target_xyz = target_xyz
            target_func_place_init = True
        target = np.hstack(
            [
                global_target_xyz,
                (1.57, 0, -1.57),
            ]
        )
        return target

    def target_func_release(interface):
        global global_target_xyz
        global target_func_release_init
        if not target_func_release_init:
            target_xyz = interface.get_xyz("EE")
            target_xyz[-1] = 0.2
            global_target_xyz = target_xyz
            target_func_release_init = True
        target = np.hstack(
            [
                global_target_xyz,
                (1.57, 0, -1.57),
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

    def gripper_control_func_close(u, gripper):
        gripper.set_gripper_status(-1)
        u[-1] = gripper.get_gripper_status()
        return u

    def gripper_control_func_open(u, gripper):
        gripper.set_gripper_status(1)
        u[-1] = gripper.get_gripper_status()
        return u

    # Randomly place the target object
    def gen_target(interface):


        def l2(pos1, pos2):
            assert len(pos1) == len(pos2)
            d = 0
            for i in range(len(pos1)):
                d = d + (pos1[i] - pos2[i]) ** 2
            d = d ** (1 / 2)
            return d

        location_boxes = np.arange(len(targets))
        np.random.shuffle(location_boxes)
        location_boxes_locations = [
            #(x1, x2, y1, y2)
            (0.43, 0.28, 0.15, 0.30),
            (0.30, 0.10, 0.25, 0.40),
            (0.12, -0.02, 0.25, 0.40),
            (0.02, -0.12, 0.25, 0.40),
            (-0.10, -0.30, 0.25, 0.40),
            (-0.28, -0.43, 0.15, 0.30),
        ]
        point_list = []
        print(interface.sim.data.qpos)
        init_positions = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.05, 1., 0., 0., 0., -0.1, 0.5, 0.03, 1., 0., 0., 0., 0.3, 0.5, 0.065, 1., 0., 0., 0., 0.4, 0.5, 0.04, 1., 0., 0., 0., 0.6, 0.5, 0.04, 1., 0., 0., 0., -0.2, 0.5, 0.06, 1., 0., 0., 0.]

        for i in range(len(interface.sim.data.qvel)):
            interface.sim.data.qvel[i] = 0.

        for i in range(len(targets)):
            while True:
                location_idx = location_boxes[i]
                # x = (np.random.rand(1) - 0.5) * (location_boxes_locations[location_idx][0] - location_boxes_locations[location_idx][1]) + (location_boxes_locations[location_idx][0] + location_boxes_locations[location_idx][1]) / 2
                # y = (np.random.rand(1) - 0.5) *(location_boxes_locations[location_idx][2] - location_boxes_locations[location_idx][3]) + (location_boxes_locations[location_idx][2] + location_boxes_locations[location_idx][3]) / 2

                location_idx = location_boxes[i]
                x = (np.random.rand(1) - 0.5) * (location_boxes_locations[i][0] - location_boxes_locations[i][1]) + (location_boxes_locations[i][0] + location_boxes_locations[i][1]) / 2
                y = (np.random.rand(1) - 0.5) *(location_boxes_locations[i][2] - location_boxes_locations[i][3]) + (location_boxes_locations[i][2] + location_boxes_locations[i][3]) / 2
                too_close = False
                for j in range(len(point_list)):
                    if l2(point_list[j], (x, y)) < 0.1 or abs(point_list[j][0] - x) < 0.1:
                        too_close = True
                if not too_close:
                    point_list.append((x, y))
                    interface.sim.data.qpos[-7 - location_idx * 7] = x
                    interface.sim.data.qpos[-6 - location_idx * 7] = y
                    interface.sim.data.qpos[-5 - location_idx * 7] = init_positions[-5 - location_idx * 7]
                    interface.sim.data.qpos[-4 - location_idx * 7] = init_positions[-4 - location_idx * 7]
                    interface.sim.data.qpos[-3 - location_idx * 7] = init_positions[-3 - location_idx * 7]
                    interface.sim.data.qpos[-2 - location_idx * 7] = init_positions[-2 - location_idx * 7]
                    interface.sim.data.qpos[-1 - location_idx * 7] = init_positions[-1 - location_idx * 7]
                    break

    gen_target(interface)
    print('init completed')
    print(interface.sim.data.qpos)
    # exit()

    
    # damp the movements of the arm
    damping = Damping(robot_config, kv=10)
    # instantiate controller
    ctrlr = OSC(
        robot_config,
        kp=200,
        null_controllers=[damping],
        vmax=[0.5, 2],  # [m/s, rad/s]
        # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, True, True, True],
    )

    # ctrlr_joint = Joint(robot_config, kp=20, kv=10)
    ctrlr_joint = Joint(
        robot_config,
        kp=200,
    )

    action_inst = np.random.randint(3)
    action_inst_dict = {
        0: 'pick',
        1: 'push',
        2: 'put_down'
    }
    if action_inst == 0:
        e = Executor(interface, robot_config.START_ANGLES, -0.05)
        # e.append(MoveTo(interface, ctrlr, target_func_5, gripper_control_func, time_limit=10))
        e.append(MoveTo(interface, ctrlr, target_func, gripper_control_func_open, error_limit=0.01))
        if target_name != 'bottle' and target_name != 'milk':
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in, gripper_control_func_open, error_limit=0.01, time_limit=200))
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in, gripper_control_func_close, time_limit=20))
        elif target_name == 'bottle':
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_bottle, gripper_control_func_open, error_limit=0.01, time_limit=200))
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_bottle, gripper_control_func_close, time_limit=20))
        elif target_name == 'milk':
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_milk, gripper_control_func_open, error_limit=0.01, time_limit=200))
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_milk, gripper_control_func_close, time_limit=20))
        e.append(MoveTo(interface, ctrlr, target_func_above_target_pick_up, gripper_control_func_close, error_limit=0.01, time_limit=200))
    elif action_inst == 1:
        e = Executor(interface, robot_config.START_ANGLES, -0.05)
        e.append(MoveTo(interface, ctrlr, target_func_move_behind, gripper_control_func_close, error_limit=0.01))
        e.append(MoveTo(interface, ctrlr, target_func_6, gripper_control_func_close, time_limit=100))
    # elif action_inst == 2:
    #     e = Executor(interface, robot_config.START_ANGLES, -0.05)
    #     e.append(MoveTo(interface, ctrlr_joint, target_func_joint_from_above_init, gripper_control_func_open, time_limit=30))
    #     # e.append(MoveTo(interface, ctrlr_joint, target_func_joint_from_above, gripper_control_func_open, time_limit=30))
    #     e.append(MoveTo(interface, ctrlr, target_func_above_target_approach, gripper_control_func_open, error_limit=0.03, time_limit=200))
    #     if target_name != 'bottle' and target_name != 'milk':
    #         e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in, gripper_control_func_open, error_limit=0.01, time_limit=200))
    #         e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in, gripper_control_func_close, time_limit=25))
    #     elif target_name == 'bottle':
    #         e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_bottle, gripper_control_func_open, error_limit=0.01, time_limit=200))
    #         e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_bottle, gripper_control_func_close, time_limit=25))
    #     elif target_name == 'milk':
    #         e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_milk, gripper_control_func_open, error_limit=0.01, time_limit=200))
    #         e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_milk, gripper_control_func_close, time_limit=25))
    #     e.append(MoveTo(interface, ctrlr, target_func_above_target_pick_up, gripper_control_func_close, error_limit=0.01, time_limit=200))
    elif action_inst == 2:
        print('yeah we are here')
        e = Executor(interface, robot_config.START_ANGLES, -0.05)
        e.append(MoveTo(interface, ctrlr, target_func, gripper_control_func_open, error_limit=0.01))
        if target_name != 'bottle' and target_name != 'milk':
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in, gripper_control_func_open, error_limit=0.01, time_limit=200))
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in, gripper_control_func_close, time_limit=20))
        elif target_name == 'bottle':
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_bottle, gripper_control_func_open, error_limit=0.01, time_limit=200))
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_bottle, gripper_control_func_close, time_limit=20))
        elif target_name == 'milk':
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_milk, gripper_control_func_open, error_limit=0.01, time_limit=200))
            e.append(MoveTo(interface, ctrlr, target_func_above_target_move_in_milk, gripper_control_func_close, time_limit=20))
        e.append(MoveTo(interface, ctrlr, target_func_above_target_pick_up, gripper_control_func_close, error_limit=0.01, time_limit=200))
        
        e.append(MoveTo(interface, ctrlr, target_func_turn_over, gripper_control_func_close, time_limit=120))
        e.append(MoveTo(interface, ctrlr, target_func_place, gripper_control_func_close, error_limit=0.01))
        e.append(MoveTo(interface, ctrlr, target_func_place, gripper_control_func_open, time_limit=20))
        e.append(MoveTo(interface, ctrlr, target_func_release, gripper_control_func_open, error_limit=0.01))
    else:
        print('action_inst not available')
        exit()


    return e, target_name, action_inst_dict[action_inst]


def rollout(executor, target_name, data_dir, data_id, action_inst):
    recorder = Recorder(data_dir, data_id)
    states = []
    state, img = executor.init_env(objects_to_track=['EE', 'target2', 'coke', 'pepsi', 'milk', 'bread', 'bottle'], goal_object=target_name, action_inst=action_inst)
    # states.append(state)
    action, end = executor.plan_action(state)
    start = True
    step = 0
    recorder.record_step(img, state, step)
    while not end:
        state, img = executor.step(action)
        action, end = executor.plan_action(state)
        step += 1
        if start == False and step >= 10:
            start = True
            step = 0
        if start == True:
            # pass
            recorder.record_step(img, state, step)
    recorder.save_states()


def collect_data(data_dir, start, end):
    # Init Simulator
    robot_config = arm('franka_panda_more_objects.xml', folder='./my_models/franka_sim')

    # Connect Interface
    interface = Mujoco(robot_config, dt=0.008, visualize=True, create_offscreen_rendercontext=False)
    interface.connect(joint_names=['panda0_joint1', 'panda0_joint2', 'panda0_joint3', 'panda0_joint4', 'panda0_joint5', 'panda0_joint6', 'panda0_joint7', 'panda0_finger_joint1'], camera_id=-1)

    # Init Offscreen Camera
    offscreen = MjRenderContextOffscreen(interface.sim, 0)

    for i in range(start, end):
        # Randomly place objects. Generate the executor for the certain action
        executor, target_name, action_inst = init_env(robot_config, interface)
        executor.offscreen = offscreen
        global global_target_xyz
        global target_func_4_init
        global target_func_6_init
        global target_func_above_target_pick_up_init
        global target_func_place_init
        global target_func_release_init
        global target_func_turn_over_init
        global_target_xyz = None
        target_func_6_init = False
        target_func_4_init = False
        target_func_above_target_pick_up_init = False
        target_func_place_init = False
        target_func_release_init = False
        target_func_turn_over_init = False

        # Rollout and record
        rollout(executor, target_name, data_dir, i, action_inst)


if __name__ == '__main__':
    data_dir = './dataset'    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    collect_data(data_dir, 0, 400)



