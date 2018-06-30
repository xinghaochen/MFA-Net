import numpy as np
from numpy.matlib import repmat
import transformations as tf

np.random.seed(10000)

class Kinematic(object):
    def __init__(self):
        self.edges = np.array([[1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [1, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [1, 14],
        [14, 15],
        [15, 16],
        [16, 17],
        [1, 18],
        [18, 19],
        [19, 20],
        [20, 21]]) - 1
        self.J = 22

    def forward_kinematic(self, bone_angle, bone_lengths, global_tral, global_rot, natural_hand_joints):
        """
        # Forward kinematic algorithm to convert the angle of the bones to the joint locations
        # bone_angle: 5x4
        # bone_length: 20-D
        # global_tral: global position, 3-D
        # global_rot: global orientation, 3-D
        #   Xinghao Chen, 1 Nov, 2016
        """
        # parameters
        wrist_id = 0
        palm_id = 1
        mcp_id = np.arange(2, 19, 4)
        pip_id = np.arange(3, 20, 4)
        dip_id = np.arange(4, 21, 4)
        tip_id = np.arange(5, 22, 4)
        mcp_bone_id = np.arange(0, 17, 4)
        pip_bone_id = np.arange(1, 18, 4)
        dip_bone_id = np.arange(2, 19, 4)
        tip_bone_id = np.arange(3, 20, 4)
        joint_hg = np.ones((4, self.J))

        # param
        origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        I = tf.identity_matrix()

        # the wrist joints and 5 MCP joints are fixed.
        joint_hg[0:3, wrist_id] = natural_hand_joints[wrist_id,:].transpose()
        joint_hg[0:3, palm_id] = natural_hand_joints[palm_id,:].transpose()
        joint_hg[0:3, mcp_id] = natural_hand_joints[mcp_id,:].transpose()

        # adjust the PIP, DIP, TIP according to the bone_angle
        for idx in range(5):
            # local coordinates
            finger = np.zeros((4, 3))
            finger[3, :] = 1
            # tip
            Ttip = tf.translation_matrix([0, bone_lengths[tip_bone_id[idx]], 0])
            Rtip = tf.rotation_matrix(bone_angle[idx, 3], xaxis)
            finger[:, 2:] = np.matmul(Rtip, np.matmul(Ttip, finger[:, 2:]))
            # dip
            Tdip = tf.translation_matrix([0, bone_lengths[dip_bone_id[idx]], 0])
            Rdip = tf.rotation_matrix(bone_angle[idx, 2], xaxis)
            finger[:, 1:] = np.matmul(Rdip, np.matmul(Tdip, finger[:, 1:]))
            # pip
            Tpip = tf.translation_matrix([0, bone_lengths[pip_bone_id[idx]], 0])
            Rpip = tf.rotation_matrix(bone_angle[idx, 1], xaxis)
            finger = np.matmul(Rpip, np.matmul(Tpip, finger))
            # transform to global coordinates
            Tf = tf.translation_matrix(joint_hg[0:3, mcp_id[idx]])
            Rf = tf.rotation_matrix(bone_angle[idx, 0], zaxis)

            # if idx ~= 5
            finger = np.matmul(Tf, np.matmul(Rf, finger))
            # else
            # finger(:, 2:end) = Rf * finger(:, 2:end)
            # finger = Tf * finger
            # end
            fidx = np.array([3,4,5])+idx*4
            joint_hg[:, fidx] = finger
        # # global translation
        # Rgx = makehgtform('xrotate', global_rot(1))
        # Rgy = makehgtform('yrotate', global_rot(2))
        # Rgz = makehgtform('zrotate', global_rot(3))
        # joint_hg = Rgz * Rgy * Rgx * joint_hg
        # # global translation and rotation
        # Tgtran = makehgtform('translate', global_pos)
        # joint_hg = Tgtran * joint_hg
        joint_hg = np.matmul(global_rot, joint_hg) + repmat(np.append(global_tral, 0).reshape(4,1), 1, self.J)
        # return
        joint_hg = joint_hg.transpose()
        joint = joint_hg[:, 0:3]
        return joint

    def inverse_kinematic(self, joints, bone_lengths, natural_hand_joints):
        """
        # Inverse kinematic algorithm to convert the joint locations to the angle of the bones
        # bone_angle: 20-D
        # bone_length: 20-D
        # global_tral: global position, 3-D
        # global_rot: global orientation, 3-D
        #   Xinghao Chen, 28 Mar, 2017
        """
        # parameters
        wrist_id = 0
        palm_id = 1
        mcp_id = np.arange(2, 19, 4)
        pip_id = np.arange(3, 20, 4)
        dip_id = np.arange(4, 21, 4)
        tip_id = np.arange(5, 22, 4)
        mcp_bone_id = np.arange(0, 17, 4)
        pip_bone_id = np.arange(1, 18, 4)
        dip_bone_id = np.arange(2, 19, 4)
        tip_bone_id = np.arange(3, 20, 4)
        joint_hg = np.ones((4, self.J))
        finger_id = np.array([mcp_id, pip_id, dip_id, tip_id])

        # param
        origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        I = tf.identity_matrix()

        ## global translation and rotation
        M = tf.affine_matrix_from_points(natural_hand_joints[np.append(wrist_id, mcp_id),:].transpose(),
                                         joints[np.append(wrist_id,mcp_id),:].transpose(),
                                         shear=False, scale=False, usesvd=False)
        scale, shear, angles, trans, persp = tf.decompose_matrix(M)
        #print scale, shear, angles, trans, persp
        global_rot = tf.euler_matrix(axes='sxyz', *angles)
        global_tral = trans

        ## angle of bones
        bone_angle = np.zeros((5, 4))
        # first inverse global translation and rotation
        joint_hg = np.ones((4, self.J))
        joint_hg[0:3, :] = joints.transpose()
        global_rot_inv = np.linalg.inv(global_rot)
        joint_hg = np.matmul(global_rot_inv, joint_hg) - repmat(np.append(global_tral, 0).reshape(4,1), 1, self.J)

        for idx in range(5):
            # mcp-pip z rotation
            vec = joint_hg[0:2, dip_id[idx]] - joint_hg[0:2, mcp_id[idx]]
            theta = np.arctan(vec[0] / vec[1])
            if abs(theta) > np.pi / 2:
                theta = np.sign(theta) * abs(abs(theta) - np.pi)
            theta = -theta
            bone_angle[idx, 0] = theta
            # reverse translation and rotation
            Tf = tf.translation_matrix(-joint_hg[0:3, mcp_id[idx]])
            Rf = tf.rotation_matrix(-theta, zaxis)
            finger = joint_hg[:, finger_id[0:4, idx]]
            finger = np.matmul(Rf, np.matmul(Tf, finger))
            joint_hg[:, finger_id[0:4, idx]] = finger

            # mcp-pip x rotation
            vec = joint_hg[1:3, pip_id[idx]] - joint_hg[1:3, mcp_id[idx]]
            theta = np.arctan(vec[1] / vec[0])
            if theta > np.pi / 8:
                theta = theta - np.pi
            bone_angle[idx, 1] = theta
            # reverse translation and rotation
            Tf = tf.translation_matrix(-joint_hg[0:3, pip_id[idx]])
            Rf = tf.rotation_matrix(-theta, xaxis)
            finger = joint_hg[:, finger_id[1:4, idx]]
            finger = np.matmul(Rf, np.matmul(Tf, finger))
            joint_hg[:, finger_id[1:4, idx]] = finger

            # pip-dip x rotation
            vec = joint_hg[1:3, dip_id[idx]] - joint_hg[1:3, pip_id[idx]]
            theta = np.arctan(vec[1] / vec[0])
            if theta > np.pi / 8:
                theta = theta - np.pi
            bone_angle[idx, 2] = theta
            # reverse translation and rotation
            Tf = tf.translation_matrix(-joint_hg[0:3, dip_id[idx]])
            Rf = tf.rotation_matrix(-theta, xaxis)
            finger = joint_hg[:, finger_id[2:4, idx]]
            finger = np.matmul(Rf, np.matmul(Tf, finger))
            joint_hg[:, finger_id[2:4, idx]] = finger

            # dip-tip x rotation
            vec = joint_hg[1:3, tip_id[idx]] - joint_hg[1:3, dip_id[idx]]
            theta = np.arctan(vec[1] / vec[0])
            if theta > np.pi / 8:
                theta = theta - np.pi
            bone_angle[idx, 3] = theta

        return bone_angle, global_tral, global_rot

    def calculate_hand_parameters(self, joint):
        #function[bone_lengths, natural_hand_joints] = calculate_hand_parameters(joint)
        # Caculate the parameters of hand model
        joint21 = np.concatenate((np.reshape(joint[0, :], (1,3)), joint[2:, :]), axis=0)
        bone_vec = joint21[self.edges[:, 0],:] - joint21[self.edges[:, 1],:]
        bone_lengths = np.sqrt(np.sum(np.square(bone_vec), axis = 1))
        natural_hand_joints = joint - repmat(joint[0,:], self.J, 1)
        # print bone_lengths, natural_hand_joints
        return bone_lengths, natural_hand_joints

if __name__ == '__main__':
    print 'kinematic'
