import glfw
import numpy as np
from mujoco_py import MjRenderContextOffscreen
from skimage import io, transform
from models.backbone_rgbd_sub_attn_separate_tar2_nets import Backbone
import math
from abr_control.utils import transformations
import random


def noun_phrase_template(target_id):
    noun_phrase = {
        0: {
            'name': ['red', 'maroon'],
            'object': ['object', 'cube', 'square'],
        },
        1: {
            'name': ['red', 'coke'],
            'object': ['can', 'bottle'],
        },
        2: {
            'name': ['blue', 'pepsi'],
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

            # # Check Window
            # if self.interface.viewer.exit:
            #     glfw.destroy_window(self.interface.viewer.window)
            #     break

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

    def execute_action(self, action):
        action._set_gripper(self.gripper)
        action.execute()

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
        # rgb_img = self.offscreen.render(width=224, height=224, mode='window', camera_name='111', depth=False)
        # # rgb_img = self.offscreen.render(width=224, height=224, mode='offscreen', camera_name='111', depth=False)
        rgb_img, d_img = self.offscreen.read_pixels(224, 224)
        rgb_img = rgb_img / 255
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32)

        # d_img = self._convert_depth_to_meters(self.interface.sim, d_img)
        # d_img[d_img > 30] = 0
        # d_img = torch.tensor(d_img.copy(), dtype=torch.float32)

        # img = torch.cat((rgb_img, d_img.unsqueeze(axis=2)), axis=2)

        img = rgb_img
        return img

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def _convert_depth_to_meters(self, sim, depth):
        extent = sim.model.stat.extent
        near = sim.model.vis.map.znear * extent
        far = sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image


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


def visualize(img, ee_pos, target_pos, target_position_pred, mean, std):

    img = img[0].detach().numpy()
    ee_pos = ee_pos[0].detach().numpy() * std + mean
    target_pos = target_pos[0].detach().numpy() * std + mean

    target_pos_xy = xyz_to_xy(target_pos[:3])
    # target_pos_xy[0] = 224 - target_pos_xy[0]
    target_pos_xy[1] = 224 - target_pos_xy[1]

    target_pos_pred_xy = xyz_to_xy(target_position_pred[:3])
    # target_pos_pred_xy[0] = 224 - target_pos_pred_xy[0]
    target_pos_pred_xy[1] = 224 - target_pos_pred_xy[1]

    ee_pos_xy = xyz_to_xy(ee_pos[:3])
    # ee_pos_xy[0] = 224 - ee_pos_xy[0]
    ee_pos_xy[1] = 224 - ee_pos_xy[1]

    # print(target_position_pred)
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img[::-1, :, :])
    circle = plt.Circle(target_pos_xy, 5, color='r')
    ax.add_patch(circle)
    circle = plt.Circle(target_pos_pred_xy, 5, color='g')
    ax.add_patch(circle)
    circle = plt.Circle(ee_pos_xy, 5, color='b')
    ax.add_patch(circle)

    # trajectory_pred_xyz = np.transpose(trajectory_pred.detach().squeeze().numpy())[:, :3] * std + mean + np.array(interface.get_xyz('EE'))

    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # x_ee = trajectory_pred[0, 0].detach().cpu().numpy()
    # y_ee = trajectory_pred[0, 1].detach().cpu().numpy()
    # z_ee = trajectory_pred[0, 2].detach().cpu().numpy()
    # x_target_gt = (target_pos[:, :3] - ee_pos[:,:3])[0, 0].detach().cpu().numpy()
    # y_target_gt = (target_pos[:, :3] - ee_pos[:,:3])[0, 1].detach().cpu().numpy()
    # z_target_gt = (target_pos[:, :3] - ee_pos[:,:3])[0, 2].detach().cpu().numpy()
    # trajectory_pred = trajectory_pred + ee_pos[:, :3]
    # print(ee_pos[0][0])
    # x_ee = trajectory_pred_xyz[:, 0] * std[0] + np.array(interface.get_xyz('EE'))[0]
    # y_ee = trajectory_pred_xyz[:, 1] * std[1] + np.array(interface.get_xyz('EE'))[1]
    # z_ee = trajectory_pred_xyz[:, 2] * std[2] + np.array(interface.get_xyz('EE'))[2]

    # # print('tar_pos normalized', target_pos)
    # # target_pos = target_pos * std + mean + np.array(interface.get_xyz('EE'))
    # # print('tar_pos recovered', target_pos)
    # x_target_gt = (target_pos[:, :3] - ee_pos[:,:3])[0, 0].detach().cpu().numpy() * std[0] + np.array(interface.get_xyz('EE'))[0]
    # y_target_gt = (target_pos[:, :3] - ee_pos[:,:3])[0, 1].detach().cpu().numpy() * std[1] + np.array(interface.get_xyz('EE'))[1]
    # z_target_gt = (target_pos[:, :3] - ee_pos[:,:3])[0, 2].detach().cpu().numpy() * std[2] + np.array(interface.get_xyz('EE'))[2]
    # ax.scatter3D(x_ee, y_ee, z_ee, c=np.arange(x_ee.shape[0]), cmap='Greens_r')
    # ax.scatter3D(x_target_gt, y_target_gt, z_target_gt, color='red')

    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(traj_gripper)
    plt.show()

    # print(x_target_gt, y_target_gt, z_target_gt, np.array(interface.get_xyz(target_name)), target_pos, target_pos * std + mean)
    return


def get_joint_angles(interface, mean_joints, std_joints):
    q = interface.get_feedback()['q']
    return torch.tensor((q - mean_joints) / std_joints, dtype=torch.float32).unsqueeze(0)



def get_inputs(target_name, interface, executor, mean, std, mean_joints, std_joints, num_traces_out=10, frames_pred=120):
    ee_pos = get_pos('EE', interface, mean, std)
    target_pos = get_pos(target_name, interface, mean, std)
    phis = torch.tensor(np.linspace(0.0, 1.0, frames_pred, dtype=np.float32)).unsqueeze(0).unsqueeze(0).repeat(1, num_traces_out, 1)
    img = executor.render_img().unsqueeze(0)
    joint_angles = get_joint_angles(interface, mean_joints, std_joints)
    return ee_pos, target_pos, phis, img, joint_angles


# def rpy2rrppyy(self, rpy):
#     rrppyy = [0] * 6
#     for i in range(3):
#         rrppyy[i * 2] = np.sin(rpy[i])
#         rrppyy[i * 2 + 1] = np.cos(rpy[i])
#     return rrppyy
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
    # print(rrppyy)
    # print(rpy)
    return rpy


def form_predictions(target_position_pred, trajectory_pred, ee_pos, std, mean, std_displacement, std_gripper, mean_gripper, t_interval):
    target_position_pred = (target_position_pred.detach().numpy() * std + mean)[0]
    ee_pos = (ee_pos.detach().numpy() * std + mean)[0]
    # ee_pos = (ee_pos.detach().numpy())[0]

    trajectory_pred = trajectory_pred[0]
    trajectory_pred_xyz = np.transpose(trajectory_pred.detach().squeeze().numpy()[:9, :]) * std + mean
    # print('traj pred raw', trajectory_pred[:, 0])
    # print('traj pred unnorm', trajectory_pred_xyz[0])
    # trajectory_pred_xyz = trajectory_pred_xyz + ee_pos
    # print('traj pred abs', trajectory_pred_xyz)
    # print('ee pos', ee_pos)

    trajectory_pred_gripper = np.transpose(trajectory_pred.detach().squeeze().numpy()[9:, :]) * std_gripper + mean_gripper

    next_xyzrpy = trajectory_pred_xyz[t_interval]
    next_gripper = trajectory_pred_gripper[t_interval]

    traj_xyz = np.repeat(np.expand_dims(next_xyzrpy[:3], axis=0), t_interval, axis=0)
    traj_rpy = np.repeat(np.expand_dims(rrppyy2rpy(next_xyzrpy[3:]), axis=0), t_interval, axis=0)
    traj_gripper = np.repeat(np.expand_dims(next_gripper, axis=0), t_interval, axis=0)

    return target_position_pred, traj_xyz, traj_rpy, traj_gripper

if __name__ == '__main__':

    import numpy as np
    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm

    # def target_func(interface):
    #     target_xyz = interface.get_xyz(target_name)
    #     target_xyz[-2] -= 0.15
    #     target = np.hstack(
    #         [
    #             target_xyz,
    #             (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation(target_name), "rxyz")[-1] + 1.57),
    #         ]
    #     )
    #     return target

    def gripper_control_func(u, gripper):
        u[-1] = gripper.get_gripper_status()
        return u

    # Randomly place the target object
    def gen_target(interface):
        
        targets = ['target2', 'coke', 'pepsi', 'milk', 'bread', 'bottle']
        target_name = targets[np.random.randint(len(targets))]


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
        # print(interface.sim.data.qpos)
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

    def xyz_to_xy(xyz):
        weight = np.array([[-1.3790e+02,  1.0139e+01,  1.8242e+00],
            [ 1.1624e-01, -9.2316e+01,  1.1633e+02]]).T
        bias = np.array([107.8063, 114.5833])
        xy = np.dot(xyz, weight) + bias
        return xy

    # sentence = input('please input instruction: ')
    # sentence = 'place down green beer bottle'
    # sentence = 'place down paper carton'
    # sentence = 'place down pepsi can'
    # sentence = 'place down coke can'
    # sentence = 'put down red cube'
    # sentence = 'pick up from above the red cube'
    # sentence = 'pick up bread'
    # sentence = 'put down bread'
    # sentence = 'pick up red cube'

    sentences = [
        # 'grab the loaf',
        # 'put down the sierra mist',
        # 'lay down the red block',
        # 'tip over the azure can',
        # 'lift the white carton',
        # 'knock over the pastry',
        # 'lift the coke can',
        # 'put down the sprite',
        # 'grab the pepsi',
        # 'elevate the red cube',
        # 'pick up the red cube',
        # 'lift up the blue cylinder',
        # 'move away the brown object',
        # 'push away the white object',
        # 'lift the blue object',
        # 'push the green sprite',
        # 'put down the green sprite',
        # 'push the reddish can',
        # 'pick up the milk container',
        # 'hold up the milk carton',
        # 'please pick up the green thing',
        # 'lift the red colored coken can',
        # 'push the yellow bread',
        # 'grab the blue colored can',
        # 'nudge that green bottle',
        # 'put down the red colored cuboid',
        # 'lift the white box',
        # 'take away the pepsi from the table',
        # 'put the green obejct forward a bit',
        # 'put down the zero coke on the desk'

        # 'put down the lime soda',
        # 'nudge that green bottle'
        # 'lift the white box',
        # 'take the pepsi off the table',
        # 'push the green object forward'
    ]
    # text_file = 't5_results_2.txt'
    # text_file = open(text_file)
    # sentences = text_file.readlines()

    import torch
    import matplotlib.pyplot as plt
    model = Backbone(img_size=224, embedding_size=192, num_traces_in=7, num_traces_out=10, num_weight_points=12, input_nc=3, device='cpu')
    model.load_state_dict(torch.load('/share/yzhou298/ckpts/extended_modattn/put_right_to/train-rgb-sub-attn-abs-action-corrected-sentence-separate-tar2-nets/1040000.pth')['model'], strict=True)
    # model.load_state_dict(torch.load('/data/Documents/yzhou298/ckpts/ckpts_from_gcp/UR5/train-12-rgb-sub-attn-fast-gripper-abs-action-take3/220000.pth')['model'], strict=True)
    model.eval()


    # mean = np.array([2.53345831e-01, 1.14758266e-01, -6.98193015e-02, 0, 0, 0, 0, 0, 0])
    # std = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])
    mean = np.array([ 2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0])
    std = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1]) ** (1/2)
    
    mean_gripper = np.array([2.12295943e-01])
    std_gripper = np.array([5.66411791e-02]) ** (1/2)

    mean_joints = np.array([-2.26736831e-01, 5.13238925e-01, -1.84928474e+00, 7.77270127e-01, 1.34229937e+00, 1.39107280e-03, 2.12295943e-01])
    std_joints = np.array([1.41245676e-01, 3.07248648e-02, 1.34113984e-01, 6.87947763e-02, 1.41992804e-01, 7.84910314e-05, 5.66411791e-02]) ** (1/2)

    std_displacement = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])


    # create our Mujoco interface
    robot_config = arm('ur5.xml', folder='./my_models/ur5_robotiq85_more_objs')
    interface = Mujoco(robot_config, dt=0.008, visualize=True)
    interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)

    # exit()
    
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

    # Example of live feed
    e = Executor(interface, robot_config.START_ANGLES, -0.05)
    e.offscreen = MjRenderContextOffscreen(e.interface.sim, 0)
    # e.offscreen = e.interface.sim

    # Run the model
    import clip


    # for sentence in sentences:
    while True:
        # sentence = input('please input instruction: ')
        target_1, target_2 = random.sample(list(np.arange(6)), 2)
        target_1 = noun_phrase_template(target_1)
        target_2 = noun_phrase_template(target_2)

        sentence = f'put {target_1} right to {target_2}'
        sentence = sentence.strip()
        print('your instruction is:', sentence)
        sentence = clip.tokenize([sentence])
        gen_target(interface)
        interface.send_target_angles(robot_config.START_ANGLES)
        # print(robot_config.START_ANGLES)
        # print(interface.sim.data.qpos)
        t_lim = 400

        t_interval = 15
        i = 0
        num_decision_steps = math.ceil(t_lim / t_interval)
        decision_step_idx = 0
        while i < t_lim:
            # Provide input from simulator
            ee_pos, target_pos, phis, img, joint_angles = get_inputs('target2', interface, e, mean, std, mean_joints, std_joints)

            # Run the model
            target_1_position_pred, target_2_position_pred, ee_pos_pred, displacement_1_pred, displacement_2_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, phis, stage=2)            # print(trajectory_pred.shape)
            # print(trajectory_pred[0, :, 0])
            
            # Extract predictions
            target_position_pred, traj_xyz, traj_rpy, traj_gripper = form_predictions(
                target_1_position_pred, trajectory_pred, ee_pos_pred, 
                std, mean, std_displacement, std_gripper, mean_gripper, t_interval)
            # print(traj_xyz)

            # Execute the predictions
            traj_obj = Trajectory(traj_xyz, traj_rpy)
            traj_obj_gripper = Trajectory_Gripper(traj_gripper)
            e.execute_action(MoveTo(interface, ctrlr, traj_obj.get_traj, traj_obj_gripper.get_traj_gripper, time_limit=45))
            
            # Visualize
            # visualize(img, ee_pos, target_pos, target_position_pred, mean, std)
            # print(target_pos[0].detach().numpy() * std + mean, target_position_pred)
            # print('orientation', transformations.euler_from_quaternion(interface.get_orientation(target_name), 'rxyz'))


            # Count steps
            i += t_interval
            decision_step_idx += 1
            # print(decision_step_idx)
            # plt.show()
        input('press any key to continue')

# 25 / 125 works for put down
# 18 / 125 works for put down