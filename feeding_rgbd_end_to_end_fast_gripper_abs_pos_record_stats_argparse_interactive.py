import glfw
import numpy as np
import mujoco_py
from mujoco_py import MjRenderContextOffscreen, MjRenderContext
from skimage import io, transform
from models.backbone_rgbd_sub_attn import Backbone
import math
import numpy as np
from abr_control.controllers import Damping
from my_osc import OSC
from mujoco_interface import Mujoco
from abr_control.utils import transformations
from my_mujoco_config import MujocoConfig as arm
import os
import torch
import matplotlib.pyplot as plt
import json
import clip
import argparse


mean = np.array([ 2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0])
std = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1]) ** (1/2)

mean_gripper = np.array([2.12295943e-01])
std_gripper = np.array([5.66411791e-02]) ** (1/2)

mean_joints = np.array([-2.26736831e-01, 5.13238925e-01, -1.84928474e+00, 7.77270127e-01, 1.34229937e+00, 1.39107280e-03, 2.12295943e-01])
std_joints = np.array([1.41245676e-01, 3.07248648e-02, 1.34113984e-01, 6.87947763e-02, 1.41992804e-01, 7.84910314e-05, 5.66411791e-02]) ** (1/2)

std_displacement = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])
mean_displacement = np.array([2.53345831e-01, 1.14758266e-01, -6.98193015e-02, 0, 0, 0, 0, 0, 0])

idx_to_name = {
    0: 'milk',
    1: 'pepsi',
    2: 'coke',
    3: 'bottle',
    4: 'bread',
    5: 'target2',
}


target_name_to_idx = {
    'target2': 0,
    'coke': 1,
    'pepsi': 2,
    'milk': 3,
    'bread': 4,
    'bottle': 5,
}


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

    def execute(self, success_criterion_fn):
        time_step = 0
        traj_xyz = []
        traj_joints = []
        success = False
        target_obj_init_xyz = None
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
                # self.gripper_control_func(self.interface)

            # send forces into Mujoco, step the sim forward
            self.interface.send_forces(u)

            # calculate time step
            time_step += 1
            # calculate error
            # ee_xyz = robot_config.Tx("EE", q=feedback["q"])
            ee_xyz = self.interface.get_xyz("EE")
            error = np.linalg.norm(ee_xyz - target[:3])

            traj_xyz.append(self.interface.get_xyz("EE"))
            traj_joints.append(self.interface.sim.data.qpos[:7])

            # whether stop criterion has been reached
            if success_criterion_fn(interface):
                success = True
            if self.time_limit is not None:
                if time_step >= self.time_limit:
                    break
            if self.error_limit is not None:
                if error <= self.error_limit:
                    break



        return traj_xyz, traj_joints, success
        

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

    def execute_action(self, action, success_criterion_fn):
        action._set_gripper(self.gripper)
        traj_xyz, traj_joints, success = action.execute(success_criterion_fn)
        return traj_xyz, traj_joints, success

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
        return state

    def render_img(self):
        self.offscreen.render(224, 224, 0)
        rgb_img, d_img = self.offscreen.read_pixels(224, 224)
        rgb_img = rgb_img / 255
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32)

        # d_img = self._convert_depth_to_meters(self.interface.sim, d_img)
        # d_img[d_img > 30] = 0
        # d_img = torch.tensor(d_img.copy(), dtype=torch.float32)

        # img = torch.cat((rgb_img, d_img.unsqueeze(axis=2)), axis=2)

        img = rgb_img
        return img


class Trajectory:
    def __init__(self, traj_xyz, traj_rpy=None):
        # traj: (seq_len, 3)
        assert traj_xyz.shape[1] == 3
        self.traj_xyz = traj_xyz
        self.traj_rpy = traj_rpy
        self.idx = 0
    
    def get_traj(self, a):
        if self.traj_rpy is None:
            to_return = np.concatenate((self.traj_xyz[self.idx], np.array([3.14, 0, 1.57])), axis=0)
        else:
            to_return = np.concatenate((self.traj_xyz[self.idx], self.traj_rpy[self.idx]), axis=0)
        # to_return[1] += 0.01
        self.idx += 1
        if self.idx == self.traj_xyz.shape[0]:
            self.idx -= 1
        return to_return


class Trajectory_Gripper:
    def __init__(self, traj):
        # traj: (seq_len, 1)
        assert traj.shape[1] == 1
        self.traj = traj
        self.idx = 0
    
    def get_traj_gripper(self, u, gripper):
        to_return = self.traj[max(self.idx, 0)]
        # to_return = self.traj[self.idx]
        if to_return > 0.3:
            to_return = 0.2
        else:
            to_return = -0.1
        self.idx += 1
        if self.idx == self.traj.shape[0]:
            self.idx -= 1
        u[-1] = to_return
        return u

    def set_traj_gripper(self, interface):
        interface.sim.data.qpos[6] = self.traj[self.idx]
        self.idx += 1
        if self.idx == self.traj.shape[0]:
            self.idx -= 1
        return


def rpy2rrppyy(rpy):
    rrppyy = [0] * 6
    for i in range(3):
        rrppyy[i * 2] = np.sin(rpy[i])
        rrppyy[i * 2 + 1] = np.cos(rpy[i])
    return rrppyy


def get_pos(name, interface, mean, std):
    xyzrryypp = np.concatenate((interface.get_xyz(name), rpy2rrppyy(transformations.euler_from_quaternion(interface.get_orientation(name), "rxyz"))))
    pos = torch.tensor((xyzrryypp - mean) / std, dtype=torch.float32).unsqueeze(0)
    return pos


def load_object_states(interface, states_dict):
    init_positions = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.05, 1., 0., 0., 0., -0.1, 0.5, 0.03, 1., 0., 0., 0., 0.3, 0.5, 0.065, 1., 0., 0., 0., 0.4, 0.5, 0.04, 1., 0., 0., 0., 0.6, 0.5, 0.04, 1., 0., 0., 0., -0.2, 0.5, 0.06, 1., 0., 0., 0.]

    for location_idx in range(6):
        xyz = states_dict[0]['objects_to_track'][idx_to_name[location_idx]]['xyz']
        print(f'gt: {idx_to_name[location_idx]}, {xyz}')
        interface.sim.data.qpos[-7 - location_idx * 7] = xyz[0]
        interface.sim.data.qpos[-6 - location_idx * 7] = xyz[1]
        interface.sim.data.qpos[-5 - location_idx * 7] = xyz[2]
        interface.sim.data.qpos[-4 - location_idx * 7] = init_positions[-4 - location_idx * 7]
        interface.sim.data.qpos[-3 - location_idx * 7] = init_positions[-3 - location_idx * 7]
        interface.sim.data.qpos[-2 - location_idx * 7] = init_positions[-2 - location_idx * 7]
        interface.sim.data.qpos[-1 - location_idx * 7] = init_positions[-1 - location_idx * 7]



def load_simulator_states(interface, robot_config, states_dict):
    load_object_states(interface, states_dict)
    for i in range(len(interface.sim.data.qvel)):
        interface.sim.data.qvel[i] = 0.
    interface.send_target_angles(robot_config.START_ANGLES)
    # print(robot_config.START_ANGLES)
    # print(interface.sim.data.qpos)
    for i in range(6):
        print('sim:', idx_to_name[i], interface.get_xyz(idx_to_name[i]))


def load_gt_traj(states_dict):
    traj_xyz_gt = np.asarray([states_dict[i]['objects_to_track']['EE']['xyz'] for i in range(len(states_dict))])
    traj_joints_gt = np.asarray([states_dict[i]['q'] for i in range(len(states_dict))])
    goal_object = states_dict[0]['goal_object']
    action_inst = states_dict[0]['action_inst']
    tar_pos = np.asarray(states_dict[0]['objects_to_track'][goal_object]['xyz'])
    return traj_xyz_gt, traj_joints_gt, goal_object, action_inst, tar_pos


action_inst_to_verb = {
    'push': ['push', 'move'],
    'pick': ['pick', 'pick up', 'raise', 'hold'],
    'put_down': ['put down', 'place down']
}

def noun_phrase_template(target_id):
    noun_phrase = {
        0: {
            'name': ['red', 'maroon'],
            'object': ['object', 'cube', 'square'],
        },
        1: {
            'name': ['red', 'coke', 'cocacola'],
            'object': ['can', 'bottle'],
        },
        2: {
            'name': ['blue', 'pepsi', 'pepsi coke'],
            'object': ['can', 'bottle'],
        },
        3: {
            'name': ['milk', 'white'],
            'object': ['carton', 'box'],
        },
        4: {
            'name': ['bread', 'yellow object', 'brown object'],
            'object': [''],
        },
        5: {
            'name': ['green', '', 'glass', 'green glass'],
            'object': ['bottle'],
        }
    }
    id_name = np.random.randint(len(noun_phrase[target_id]['name']))
    id_object = np.random.randint(len(noun_phrase[target_id]['object']))
    name = noun_phrase[target_id]['name'][id_name]
    obj = noun_phrase[target_id]['object'][id_object]
    return (name + ' ' + obj).strip()

def verb_phrase_template(action_inst):
    if action_inst is None:
        action_inst = random.choice(list(action_inst_to_verb.keys()))
    action_id = np.random.randint(len(action_inst_to_verb[action_inst]))
    verb = action_inst_to_verb[action_inst][action_id]
    return verb.strip()

def sentence_template(target_id, action_inst=None):
    sentence = ''
    verb = verb_phrase_template(action_inst)
    sentence = sentence + verb
    sentence = sentence + ' ' + noun_phrase_template(target_id)
    return sentence.strip()


class PutDownSuccess:
    def __init__(self, interface, goal_object):
        self.pick_success = False
        self.put_success = False
        self.rotated = False
        self.dropped = False
        self.target_obj_init_xyz = interface.get_xyz(goal_object)
        self.goal_object = goal_object

    def __call__(self, interface):

        print(self.pick_success, self.put_success, self.rotated, self.dropped)

        if not self.pick_success:
            target_obj_curr_xyz = interface.get_xyz(self.goal_object)
            self.pick_success = (target_obj_curr_xyz[2] > 0.1)

        if self.pick_success:
            # whether object rotated
            target_orientation = (-1.57, 0, 0)
            current_rpy = transformations.euler_from_quaternion(interface.get_orientation(self.goal_object), 'rxyz')

            if l2(current_rpy, target_orientation) < 0.5:
                self.rotated = True

            # whether object dropped
            target_obj_curr_xyz = interface.get_xyz(self.goal_object)
            if (target_obj_curr_xyz[2] < 0.08):
                self.dropped = True

            if self.rotated and self.dropped:
                self.put_success = True

        if self.pick_success and self.put_success:
            return True


def success_criterion_fn(interface, action_inst, goal_object):
    assert action_inst in ['pick', 'push', 'put_down']
    target_obj_init_xyz = interface.get_xyz(goal_object)
    
    def pick_success(interface):
        target_obj_curr_xyz = interface.get_xyz(goal_object)
        return target_obj_curr_xyz[2] - target_obj_init_xyz[2] > 0.03

    def push_success(interface):
        target_obj_curr_xyz = interface.get_xyz(goal_object)
        return target_obj_curr_xyz[1] - target_obj_init_xyz[1] > 0.03

    if action_inst == 'pick':
        return pick_success
    elif action_inst == 'push':
        return push_success
    elif action_inst == 'put_down':
        return PutDownSuccess(interface, goal_object)


def get_joint_angles(interface, mean_joints, std_joints):
    q = interface.get_feedback()['q']
    return torch.tensor((q - mean_joints) / std_joints, dtype=torch.float32).unsqueeze(0)


def get_inputs(target_name, interface, executor, mean, std, mean_joints, std_joints, num_traces_out=10, frames_pred=120):
    ee_pos = get_pos('EE', interface, mean, std)
    print(target_name)
    target_pos = get_pos(target_name, interface, mean, std)
    phis = torch.tensor(np.linspace(0.0, 1.0, frames_pred, dtype=np.float32)).unsqueeze(0).unsqueeze(0).repeat(1, num_traces_out, 1)
    img = executor.render_img().unsqueeze(0)
    joint_angles = get_joint_angles(interface, mean_joints, std_joints)
    return ee_pos, target_pos, phis, img, joint_angles


def rrppyy2rpy(rrppyy):
    rpy = [0] * 3
    for i in range(3):
        tan_theta = rrppyy[i * 2] / rrppyy[i * 2 + 1]
        theta = np.arctan(tan_theta)
        if theta >= 0:
            if rrppyy[i * 2] < 0:
                theta = theta + np.pi
        elif theta < 0:
            if rrppyy[i * 2] < 0:
                theta = theta + np.pi * 2
            elif rrppyy[i * 2] > 0:
                theta = theta + np.pi
        rpy[i] = theta

    return rpy


def form_predictions(target_position_pred, trajectory_pred, ee_pos, std, mean, std_displacement, std_gripper, mean_gripper, t_interval):
    target_position_pred = (target_position_pred.detach().numpy() * std + mean)[0]
    ee_pos = (ee_pos.detach().numpy() * std + mean)[0]

    trajectory_pred = trajectory_pred[0]
    trajectory_pred_xyz = np.transpose(trajectory_pred.detach().squeeze().numpy()[:9, :]) * std + mean

    trajectory_pred_gripper = np.transpose(trajectory_pred.detach().squeeze().numpy()[9:, :]) * std_gripper + mean_gripper

    next_xyzrpy = trajectory_pred_xyz[t_interval]
    next_gripper = trajectory_pred_gripper[t_interval]

    traj_xyz = np.repeat(np.expand_dims(next_xyzrpy[:3], axis=0), t_interval, axis=0)
    traj_rpy = np.repeat(np.expand_dims(rrppyy2rpy(next_xyzrpy[3:]), axis=0), t_interval, axis=0)
    traj_gripper = np.repeat(np.expand_dims(next_gripper, axis=0), t_interval, axis=0)

    return target_position_pred, traj_xyz, traj_rpy, traj_gripper


def run_simulator(model, interface, executor, ctrlr, goal_object, action_inst):

    t_lim = 400
    t_interval = 15
    i = 0
    num_decision_steps = math.ceil(t_lim / t_interval)
    decision_step_idx = 0

    # sentence = sentence_template(target_name_to_idx[goal_object], action_inst)
    # print(sentence)
    # sentence = clip.tokenize([sentence])
    sentence = None
    success_fn = success_criterion_fn(interface, action_inst, goal_object)
    success = False

    stats = {
        'ee_gt': [],
        'ee_pred': [],
        'tar_gt': [],
        'tar_pred':[],
        'disp_gt':[],
        'disp_pred':[]
    }

    while i < t_lim:
        # Provide input from simulator
        ee_pos, target_pos, phis, img, joint_angles = get_inputs(goal_object, interface, executor, mean, std, mean_joints, std_joints)

        if sentence is None:
            sentence = input('please input instruction: ')
            sentence = sentence.strip()
            sentence = clip.tokenize([sentence])

        # Run the model
        target_position_pred, ee_pos_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, phis, stage=2)
        
        # Extract predictions
        target_position_pred, traj_xyz, traj_rpy, traj_gripper = form_predictions(
            target_position_pred, trajectory_pred, ee_pos, 
            std, mean, std_displacement, std_gripper, mean_gripper, t_interval)

        # Execute the predictions
        traj_obj = Trajectory(traj_xyz, traj_rpy)
        traj_obj_gripper = Trajectory_Gripper(traj_gripper)
        traj_xyz_exec, traj_joints_exec, success = executor.execute_action(
            MoveTo(interface, ctrlr, traj_obj.get_traj, traj_obj_gripper.get_traj_gripper, time_limit=45), 
            success_fn)

        # Visualize
        # visualize(img, ee_pos, target_pos, target_position_pred, mean, std)
        # print(target_pos[0].detach().numpy() * std + mean, target_position_pred)
        # print('orientation', transformations.euler_from_quaternion(interface.get_orientation(target_name), 'rxyz'))
        print('success', success)

        # if success:
        #     break

        # Count steps
        i += t_interval
        decision_step_idx += 1
        print(decision_step_idx)

        ee_pos = ee_pos.detach().numpy() * std + mean
        ee_pos_pred = ee_pos_pred.detach().numpy() * std + mean
        target_pos = target_pos.detach().numpy() * std + mean
        stats['ee_gt'].append(ee_pos)
        stats['ee_pred'].append(ee_pos_pred)
        stats['tar_gt'].append(target_pos)
        stats['tar_pred'].append(np.expand_dims(target_position_pred, axis=0))
        stats['disp_gt'].append(target_pos - ee_pos)
        stats['disp_pred'].append(displacement_pred.detach().numpy() * std_displacement + mean_displacement)

    return stats, success


    # # Prepare inputs
    # sentence = sentence_template(target_name_to_idx[goal_object], action_inst)
    # print(f'language input:', sentence)
    # sentence = clip.tokenize([sentence])
    # img = torch.tensor(executor.render_img() / 255, dtype=torch.float32).unsqueeze(0)
    # ee_pos = torch.tensor((interface.get_xyz('EE') - mean) / std, dtype=torch.float32).unsqueeze(0)
    # phis = torch.tensor(phis, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # target_position_pred, attn_map, attn_map2 = model_perception(img, sentence)
    # target_position_pred = target_position_pred * torch.tensor(std, dtype=torch.float32)
    # trajectory_pred, dmp_weights = model(ee_pos[:, :3], target_position_pred[:, :3], phis, sentence)

    # target_position_pred = target_position_pred.detach().numpy() * std + mean
    # traj_xyz = np.transpose(trajectory_pred.detach().squeeze().numpy())[:, :3] * std + mean
    # traj_gripper = np.transpose(trajectory_pred.detach().squeeze().numpy())[:, 3:] * std_gripper + mean_gripper

    # traj_obj = Trajectory(traj_xyz)
    # traj_obj_gripper = Trajectory_Gripper(traj_gripper)
    # traj_xyz_exec, traj_joints_exec, success = executor.execute_action(MoveTo(interface, ctrlr, traj_obj.get_traj, traj_obj_gripper.get_traj_gripper, time_limit=phis.shape[-1]), success_criterion_fn(interface, action_inst, goal_object))

    # # Post process
    # traj_xyz_exec = np.array(traj_xyz_exec)
    # traj_joints_exec = np.array(traj_joints_exec)

    # print('success:', success)

    # return target_position_pred, traj_xyz_exec, traj_joints_exec, success


def l2(xyz1, xyz2):
    squared_sum = (xyz1[0] - xyz2[0]) ** 2 + (xyz1[1] - xyz2[1]) ** 2 + (xyz1[2] - xyz2[2]) ** 2
    return math.sqrt(squared_sum)


def l2_traj(traj1, traj2):
    assert traj2.shape[0] == traj1.shape[0] * 3
    num_points = traj1.shape[0]
    width = traj1.shape[1]
    sum_error = 0
    for i in range(num_points):
        i_traj2 = 3 * i + 1
        error_this_time = 0
        for j in range(width):
            error_this_time += (traj1[i][j] - traj2[i_traj2][j]) ** 2
        error_this_time = math.sqrt(error_this_time)
        sum_error += error_this_time
    return sum_error


def l1_traj(traj1, traj2):
    assert traj2.shape[0] == traj1.shape[0] * 3
    num_points = traj1.shape[0]
    width = traj1.shape[1]
    sum_error = 0
    for i in range(num_points):
        i_traj2 = 3 * i + 1
        error_this_time = 0
        for j in range(width):
            error_this_time += abs(traj1[i][j] - traj2[i_traj2][j])
        sum_error += (error_this_time / width)
    return sum_error


def l2_seq(a, b, dim=3):
    a = np.concatenate(a, axis=0)
    b = np.concatenate(b, axis=0)
    diff = a - b
    err = np.sqrt(np.sum(diff[:, :3] * diff[:, :3], axis=1))
    return np.mean(err)


def calculate_error(stats):
    tar_pos_error = l2_seq(stats['tar_pred'], stats['tar_gt'])
    ee_pos_error = l2_seq(stats['ee_pred'], stats['ee_gt'])
    disp_error = l2_seq(stats['disp_pred'], stats['disp_gt'])
    return tar_pos_error, ee_pos_error, disp_error

def test_1_rollout(model, interface, robot_config, executor, ctrlr, states_dict, demonstration_dir, result_file_fd):
    load_simulator_states(interface, robot_config, states_dict)
    traj_xyz_gt, traj_joints_gt, goal_object, action_inst, tar_pos_gt = load_gt_traj(states_dict)
    stats, success = run_simulator(model, interface, executor, ctrlr, goal_object, action_inst)
    # tar_pos_error, ee_pos_error, disp_error = calculate_error(stats)
    # print(tar_pos_error, traj_xyz_error, traj_joints_error)
    # result_file_fd.write(f'{demonstration_dir} {goal_object} {action_inst} {success} {tar_pos_error} {ee_pos_error} {disp_error}\n')
    # np.save(f'./traj/{demonstration_dir.split(r"/")[-1]}_ur5_traj_xyz.npy', traj_xyz_exec)
    # np.save(f'./traj/{demonstration_dir.split(r"/")[-1]}_ur5_traj_joints.npy', traj_joints_exec)
    return


def test(model, interface, robot_config, executor, ctrlr, data_dirs, result_file):
    all_dirs = []
    for data_dir in data_dirs:
        all_dirs = all_dirs + [ f.path for f in os.scandir(data_dir) if f.is_dir() ]
    for idx, demonstration_dir in enumerate(all_dirs):

        states_json = os.path.join(demonstration_dir, 'states.json')
        with open(states_json) as json_file:
            states_dict = json.load(json_file)
            json_file.close()

        print(idx, demonstration_dir)

        result_file_fd = open(result_file, 'a')
        test_1_rollout(model, interface, robot_config, executor, ctrlr, states_dict, demonstration_dir, result_file_fd)
        result_file_fd.close()
        # exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_dirs', required=True)
    parser.add_argument('--result_file', required=True)
    args = parser.parse_args()

    print(args)
    
    # Load model
    model = Backbone(img_size=224, embedding_size=192, num_traces_in=7, num_traces_out=10, num_weight_points=12, input_nc=3, device='cpu')
    model.load_state_dict(torch.load(args.ckpt)['model'], strict=True)
    model.eval()

    # Load simulator
    # create our Mujoco interface
    robot_config = arm('ur5.xml', folder='./my_models/ur5_robotiq85_more_objs')
    interface = Mujoco(robot_config, dt=0.008)
    interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)

    # damp the movements of the arm
    damping = Damping(robot_config, kv=10)
    # instantiate controller
    ctrlr = OSC(
        robot_config,
        kp=200,
        null_controllers=[damping],
        vmax=[0.5, 0.5],  # [m/s, rad/s]
        ctrlr_dof=[True, True, True, True, True, True],
    )
    # Create executor and offscreen camera
    executor = Executor(interface, robot_config.START_ANGLES, -0.05)
    executor.offscreen = MjRenderContextOffscreen(executor.interface.sim, 0)

    # Test data dir
    data_dirs = [args.data_dirs]
    
    # Result file
    result_file = args.result_file

    # Start testing
    test(model, interface, robot_config, executor, ctrlr, data_dirs, result_file)
