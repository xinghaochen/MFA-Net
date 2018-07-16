'''
Extract features from dynamic hand gesture squences
Xinghao Chen, 27 Mar, 2017
'''
import scipy.io as sio
import numpy as np
from show_3d_joints import show_3d_joints
import math
import kinematic
import matplotlib.pyplot as plt

# load data
data = sio.loadmat('../../data/DHGdata/DHGdata_testid_1.mat')
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']
# print x_train, x_test, y_train, y_test

N_train = x_train.shape[1]
N_test = x_test.shape[1]
J = 22
is_show = 1
frame_feature_dim = 25
type_names = {'train', 'test'}
type_id = 1
print N_train, N_test, x_train.shape[1], type(x_train)

# ref id to calculate the global rotation
refid = 58
frame = x_train[0, refid]
joint = frame[0, :]
joint = np.reshape(joint, (J, 3))
#print joint
K = kinematic.Kinematic()

bone_lengths, natural_hand_joints = K.calculate_hand_parameters(joint)
bone_angle = -np.random.rand(5, 4)/2 * 0
bone_angle[:, 0] = 0#bone_angle[:, 0] + 0.5
bone_angle[2, 1:] = np.array([0.0, 0.1, 0.1])*(-1)
print bone_angle
global_tral = np.zeros((3, 1))
global_rot = np.eye(4)
# bone_angle, global_tral, global_rot = K.inverse_kinematic(joint, bone_lengths, natural_hand_joints)
# print bone_angle, global_tral, global_rot
# joint_fk = K.forward_kinematic(bone_angle, bone_lengths, global_tral, global_rot, natural_hand_joints)
# show_3d_joints(joint_fk, 1, 1)
# plt.figure()
#show_3d_joints(joint_fk, 1, 1)

# extract per frame hand parameters
if type_id == 1:
    N = N_train
    x = x_train
    y = y_train
else:
    N = N_test
    x = x_test
    y = y_test

for sid in range(0, N-1): # sample id
    sid = np.random.randint(N)
    print sid
    N_frame= x[0, sid].shape[0]
    frame = x[0, sid]
    for fid in range(0,1): #N_frame# frame id
        joint = frame[fid,:]
        joint = np.reshape(joint, (J, 3))

        bone_lengths, natural_hand_joints = K.calculate_hand_parameters(joint)
        #show_3d_joints(natural_hand_joints, 1, 1)

        bone_angle, global_tral, global_rot = K.inverse_kinematic(joint, bone_lengths, natural_hand_joints)
        # [a, b, c] = cart2sph(global_tral(1),global_tral(2),global_tral(3))
        # tform2eul(global_rot)

        # display 3d joints
        if is_show:
            show_3d_joints(joint, 1, 1)
            # view([90 0])
            joint_fk = K.forward_kinematic(bone_angle, bone_lengths, global_tral, global_rot, natural_hand_joints)
            show_3d_joints(joint_fk, 1, 1)
