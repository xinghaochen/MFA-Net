import numpy as np
from numpy import linalg as la
from numpy.matlib import repmat
import kinematic
import transformations as tf
from scipy.stats import norm
import scipy.io as sio
import os
import math
from numpy.matlib import repmat

np.random.seed(10000)

def extract_feature_for_frame_vae(joint, natural_hand_joints, vae):
    """
    function feature = extract_feature_frame(joint, natural_hand_joints)
    # Extract skelton feature from joint
    # including global translation, global rotation and bone angles
    #   Xinghao Chen, 28 Mar, 2017
    """
    K = kinematic.Kinematic()
    bone_lengths, _ = K.calculate_hand_parameters(joint)
    # normalize palm bone
    for id in np.array([1, 7, 11, 15, 19])-1:
        b1 = la.norm(joint[id,:] - joint[1,:])
        b2 = la.norm(natural_hand_joints[id,:] - natural_hand_joints[1,:])
        natural_hand_joints[id,:] = natural_hand_joints[1,:] + b1 / b2 * (natural_hand_joints[id,:] - natural_hand_joints[1,:])
    b1 = la.norm(joint[2,:] - joint[0,:])
    b2 = la.norm(natural_hand_joints[2,:] - natural_hand_joints[0,:])
    natural_hand_joints[2,:] = natural_hand_joints[0,:] + b1 / b2 * (natural_hand_joints[2,:] - natural_hand_joints[0,:])

    [bone_angle, global_tral, global_rot] = K.inverse_kinematic(joint, bone_lengths, natural_hand_joints)
    a, e, r = cart2sph(global_tral[0], global_tral[1], global_tral[2])
    _, _, eul, _, _ = tf.decompose_matrix(global_rot)
    # normalize
    bone_angle[:, 1:] = bone_angle[:, 1:] + np.pi / 4
    feature_global = np.append(np.append(a, e), eul)
    feature_hand = np.reshape(bone_angle, (1, -1))
    # vae feature
    pose = joint.reshape((1,-1))
    pose = pose - repmat(pose[:, 3:6], 1, pose.shape[1]/3)
    pose = (pose/0.1 + 1)/2

    #self.vae = pose_vae.PoseVAE(dataset_='DHG2016', test_id_=test_id)
    #with vae.vae_graph.as_default():
    feature_hand = vae.encode(pose)#np.append(vae.encode(pose)*10, feature_hand)
    #print feature_hand
    feature = np.append(feature_global, feature_hand)
    # print feature_global.shape, feature_hand.shape
    return feature

def extract_feature_for_frame(joint, natural_hand_joints):
    """
    function feature = extract_feature_frame(joint, natural_hand_joints)
    # Extract skelton feature from joint
    # including global translation, global rotation and bone angles
    #   Xinghao Chen, 28 Mar, 2017
    """
    K = kinematic.Kinematic()
    bone_lengths, _ = K.calculate_hand_parameters(joint)
    # normalize palm bone
    for id in np.array([1, 7, 11, 15, 19])-1:
        b1 = la.norm(joint[id,:] - joint[1,:])
        b2 = la.norm(natural_hand_joints[id,:] - natural_hand_joints[1,:])
        natural_hand_joints[id,:] = natural_hand_joints[1,:] + b1 / b2 * (natural_hand_joints[id,:] - natural_hand_joints[1,:])
    b1 = la.norm(joint[2,:] - joint[0,:])
    b2 = la.norm(natural_hand_joints[2,:] - natural_hand_joints[0,:])
    natural_hand_joints[2,:] = natural_hand_joints[0,:] + b1 / b2 * (natural_hand_joints[2,:] - natural_hand_joints[0,:])

    [bone_angle, global_tral, global_rot] = K.inverse_kinematic(joint, bone_lengths, natural_hand_joints)
    a, e, r = cart2sph(global_tral[0], global_tral[1], global_tral[2])
    _, _, eul, _, _ = tf.decompose_matrix(global_rot)
    # normalize
    bone_angle[:, 1:] = bone_angle[:, 1:] + np.pi / 4
    feature_global = np.append(np.append(a, e), eul)
    feature_hand = np.reshape(bone_angle, (1, -1))
    feature = np.append(feature_global, feature_hand)
    return feature


def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return az, elev, r


def extract_feature_global_amp(seq, M, max_dist_factor):
    """
    function[global_amp_feature, palm_radius] = extract_feature_global_amp(seq, M, max_dist_factor)
    # Extract global translation ampitude feature from a gesture sequence
    #   Xinghao Chen,  28 Mar, 2016
    """
    #     M = 5
    #     max_dist_factor = 1.5
    palm_id = 1
    palm_xyz_id = range(3, 6)
    wrist_xyz_id = range(0, 3)
    mid_mcp_xyz_id = range(30, 33)

    N = seq.shape[0]
    global_amp_feature = seq[:, palm_xyz_id] - repmat(seq[0, palm_xyz_id], N, 1)
    global_amp_feature = np.sqrt(np.sum(np.square(global_amp_feature), axis=1))
    palm_radius = np.sqrt(np.sum(np.square(seq[:, palm_xyz_id] - seq[:, wrist_xyz_id]), axis=1))+\
                  np.sqrt(np.sum(np.square(seq[:, palm_xyz_id] - seq[:, mid_mcp_xyz_id]), axis=1))
    palm_radius = np.mean(palm_radius)
    # bin
    thres = np.zeros((M, 1))
    sigma = palm_radius
    mu = 0
    max_range_prob = norm.cdf(-sigma * max_dist_factor, loc=mu, scale=sigma)
    for k in range(M):
        prob = (1 - (1 - 2 * max_range_prob) / (M - k)) / 2
        thres[k] = -norm.ppf(prob, loc=mu, scale=sigma)
    thres = np.append([0], thres)
    # thresholding
    global_amp_feature_bin = np.zeros(global_amp_feature.shape)
    for k in range(1, M+1):
        idx = (global_amp_feature >= thres[k - 1]) & (global_amp_feature < thres[k])
        global_amp_feature_bin[idx] = k - 1
    idx = (global_amp_feature > thres[M])
    global_amp_feature_bin[idx] = M
    global_amp_feature_bin = (global_amp_feature_bin - 1) * 1.0 / M
    global_amp_feature = global_amp_feature_bin
    return global_amp_feature, palm_radius


def extract_all_feature_for_frame(x, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize):
    """
    extract motion feature from a gesture sequence
    """
    # params
    is_filter = 0
    N = x.shape[0]
    J = 22
    global_feature_dim = 6
    hand_feature_dim = 20
    frame_feature_dim = hand_feature_dim + global_feature_dim
    # load hand model parameters
    data_dir = os.path.dirname(__file__)
    if test_id == 2:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_2_187.mat')
    else:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_1_58.mat')
    dmat = sio.loadmat(filename)
    natural_hand_joints = dmat['natural_hand_joints']
    # filter
    if is_filter:
        x[2:N-2,:] = (-3 * x[0:N-4,:] + 12 * x[1:N-3,:] + 17 * x[2:N-2,:] +
                      12 * x[3:N-1,:] - 3 * x[4:,:]) / 35
    # motion feature
    gesture_feature = np.zeros((N, frame_feature_dim - 1))
    for fid in range(N):  # frame id
        joint = x[fid,:]
        joint = np.reshape(joint, (-1, 3))
        frame_feature = extract_feature_for_frame(joint, natural_hand_joints)
        gesture_feature[fid,:] = frame_feature
    # global amp feature
    [global_amp_feature, palm_radius] = extract_feature_global_amp(x, M, max_dist_factor)
    global_amp_feature = np.reshape(global_amp_feature, (-1, 1))
    # combine motion and amp feature
    gesture_feature = np.concatenate((global_amp_feature, gesture_feature), axis=1)
    # offset pose
    gesture_feature_op = gesture_feature - repmat(gesture_feature[0,:], gesture_feature.shape[0], 1)
    # dynamic pose
    gesture_feature_dp1 = gesture_feature
    gesture_feature_dp1[1:,:] = gesture_feature[1:,:] - gesture_feature[0:-1,:]
    of = offset1  # of = 5;
    gesture_feature_dp5 = gesture_feature
    gesture_feature_dp5[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    of = offset2  # of = 10;
    gesture_feature_dp10 = gesture_feature
    gesture_feature_dp10[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    # # static pose
    # global_feature_tmp = gesture_feature[:, 0:global_feature_dim]
    # global_feature_sp = np.zeros(N, global_feature_dim * (global_feature_dim - 1))
    # for k in range(global_feature_dim):
    #     global_feature_sp(:, (k - 1) * (global_feature_dim - 1) + 1:k * (global_feature_dim - 1)) = global_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(global_feature_tmp(:, k), 1, global_feature_dim - 1);
    # hand_feature_tmp = gesture_feature(:, global_feature_dim + 1:end);
    # hand_feature_sp = zeros(N_frame, hand_feature_dim * (hand_feature_dim - 1));
    # for k = 1:hand_feature_dim
    #     hand_feature_sp(:, (k - 1) * (hand_feature_dim - 1) + 1:k * (hand_feature_dim - 1)) = hand_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(hand_feature_tmp(:, k), 1, hand_feature_dim - 1);
    # concate all features
    #         fglobal = [gesture_feature(:,1:global_feature_dim), global_feature_sp, gesture_feature_op(:,1:global_feature_dim), gesture_feature_dp1(:,1:global_feature_dim), gesture_feature_dp5(:,1:global_feature_dim), gesture_feature_dp10(:,1:global_feature_dim)];
    #     fh = [gesture_feature(:,global_feature_dim+1:end), hand_feature_sp, gesture_feature_op(:,global_feature_dim+1:end), gesture_feature_dp1(:,global_feature_dim+1:end), gesture_feature_dp5(:,global_feature_dim+1:end), gesture_feature_dp10(:,global_feature_dim+1:end)];
    fglobal = np.concatenate((gesture_feature[:, 0:global_feature_dim], gesture_feature_op[:, 0:global_feature_dim],
                              gesture_feature_dp1[:, 0:global_feature_dim], gesture_feature_dp5[:, 0:global_feature_dim],
                              gesture_feature_dp10[:, 0:global_feature_dim]), axis=1)
    fhand = np.concatenate((gesture_feature[:, global_feature_dim:], gesture_feature_op[:, global_feature_dim:],
                            gesture_feature_dp1[:, global_feature_dim:], gesture_feature_dp5[:, global_feature_dim:],
                            gesture_feature_dp10[:, global_feature_dim:]), axis=1)
    feature_hand = fhand
    feature_global = fglobal
    return feature_hand, feature_global

def extract_all_feature_for_frame_vae(x, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize, vae):
    """
    extract motion feature from a gesture sequence
    """
    # params
    is_filter = 0
    N = x.shape[0]
    J = 22
    global_feature_dim = 6
    hand_feature_dim = 20
    frame_feature_dim = hand_feature_dim + global_feature_dim
    # load hand model parameters
    data_dir = os.path.dirname(__file__)
    if test_id == 2:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_2_187.mat')
    else:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_1_58.mat')
    dmat = sio.loadmat(filename)
    natural_hand_joints = dmat['natural_hand_joints']
    # filter
    if is_filter:
        x[2:N-2,:] = (-3 * x[0:N-4,:] + 12 * x[1:N-3,:] + 17 * x[2:N-2,:] +
                      12 * x[3:N-1,:] - 3 * x[4:,:]) / 35
    # motion feature
    gesture_feature = np.zeros((N, frame_feature_dim - 1))
    for fid in range(N):  # frame id
        joint = x[fid,:]
        joint = np.reshape(joint, (-1, 3))
        frame_feature = extract_feature_for_frame_vae(joint, natural_hand_joints, vae)
        gesture_feature[fid,:] = frame_feature
    # global amp feature
    [global_amp_feature, palm_radius] = extract_feature_global_amp(x, M, max_dist_factor)
    global_amp_feature = np.reshape(global_amp_feature, (-1, 1))
    # combine motion and amp feature
    gesture_feature = np.concatenate((global_amp_feature, gesture_feature), axis=1)
    # offset pose
    gesture_feature_op = gesture_feature - repmat(gesture_feature[0,:], gesture_feature.shape[0], 1)
    # dynamic pose
    gesture_feature_dp1 = gesture_feature
    gesture_feature_dp1[1:,:] = gesture_feature[1:,:] - gesture_feature[0:-1,:]
    of = offset1  # of = 5;
    gesture_feature_dp5 = gesture_feature
    gesture_feature_dp5[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    of = offset2  # of = 10;
    gesture_feature_dp10 = gesture_feature
    gesture_feature_dp10[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    # # static pose
    # global_feature_tmp = gesture_feature[:, 0:global_feature_dim]
    # global_feature_sp = np.zeros(N, global_feature_dim * (global_feature_dim - 1))
    # for k in range(global_feature_dim):
    #     global_feature_sp(:, (k - 1) * (global_feature_dim - 1) + 1:k * (global_feature_dim - 1)) = global_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(global_feature_tmp(:, k), 1, global_feature_dim - 1);
    # hand_feature_tmp = gesture_feature(:, global_feature_dim + 1:end);
    # hand_feature_sp = zeros(N_frame, hand_feature_dim * (hand_feature_dim - 1));
    # for k = 1:hand_feature_dim
    #     hand_feature_sp(:, (k - 1) * (hand_feature_dim - 1) + 1:k * (hand_feature_dim - 1)) = hand_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(hand_feature_tmp(:, k), 1, hand_feature_dim - 1);
    # concate all features
    #         fglobal = [gesture_feature(:,1:global_feature_dim), global_feature_sp, gesture_feature_op(:,1:global_feature_dim), gesture_feature_dp1(:,1:global_feature_dim), gesture_feature_dp5(:,1:global_feature_dim), gesture_feature_dp10(:,1:global_feature_dim)];
    #     fh = [gesture_feature(:,global_feature_dim+1:end), hand_feature_sp, gesture_feature_op(:,global_feature_dim+1:end), gesture_feature_dp1(:,global_feature_dim+1:end), gesture_feature_dp5(:,global_feature_dim+1:end), gesture_feature_dp10(:,global_feature_dim+1:end)];
    fglobal = np.concatenate((gesture_feature[:, 0:global_feature_dim], gesture_feature_op[:, 0:global_feature_dim],
                              gesture_feature_dp1[:, 0:global_feature_dim], gesture_feature_dp5[:, 0:global_feature_dim],
                              gesture_feature_dp10[:, 0:global_feature_dim]), axis=1)
    fhand = np.concatenate((gesture_feature[:, global_feature_dim:], gesture_feature_op[:, global_feature_dim:],
                            gesture_feature_dp1[:, global_feature_dim:], gesture_feature_dp5[:, global_feature_dim:],
                            gesture_feature_dp10[:, global_feature_dim:]), axis=1)
    feature_hand = fhand
    feature_global = fglobal
    return feature_hand, feature_global


def extract_all_feature_for_frame_vae_noamp(x, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize, vae):
    """
    extract motion feature from a gesture sequence
    """
    # params
    is_filter = 0
    N = x.shape[0]
    J = 22
    global_feature_dim = 5
    hand_feature_dim = 20
    frame_feature_dim = hand_feature_dim + global_feature_dim
    # load hand model parameters
    data_dir = os.path.dirname(__file__)
    if test_id == 2:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_2_187.mat')
    else:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_1_58.mat')
    dmat = sio.loadmat(filename)
    natural_hand_joints = dmat['natural_hand_joints']
    # filter
    if is_filter:
        x[2:N-2,:] = (-3 * x[0:N-4,:] + 12 * x[1:N-3,:] + 17 * x[2:N-2,:] +
                      12 * x[3:N-1,:] - 3 * x[4:,:]) / 35
    # motion feature
    gesture_feature = np.zeros((N, frame_feature_dim))
    for fid in range(N):  # frame id
        joint = x[fid,:]
        joint = np.reshape(joint, (-1, 3))
        frame_feature = extract_feature_for_frame_vae(joint, natural_hand_joints, vae)
        gesture_feature[fid,:] = frame_feature
    # global amp feature
    # [global_amp_feature, palm_radius] = extract_feature_global_amp(x, M, max_dist_factor)
    # global_amp_feature = np.reshape(global_amp_feature, (-1, 1))
    # combine motion and amp feature
    # gesture_feature = np.concatenate((global_amp_feature, gesture_feature), axis=1)
    # offset pose
    gesture_feature_op = gesture_feature - repmat(gesture_feature[0,:], gesture_feature.shape[0], 1)
    # dynamic pose
    gesture_feature_dp1 = gesture_feature
    gesture_feature_dp1[1:,:] = gesture_feature[1:,:] - gesture_feature[0:-1,:]
    of = offset1  # of = 5;
    gesture_feature_dp5 = gesture_feature
    gesture_feature_dp5[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    of = offset2  # of = 10;
    gesture_feature_dp10 = gesture_feature
    gesture_feature_dp10[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    # # static pose
    # global_feature_tmp = gesture_feature[:, 0:global_feature_dim]
    # global_feature_sp = np.zeros(N, global_feature_dim * (global_feature_dim - 1))
    # for k in range(global_feature_dim):
    #     global_feature_sp(:, (k - 1) * (global_feature_dim - 1) + 1:k * (global_feature_dim - 1)) = global_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(global_feature_tmp(:, k), 1, global_feature_dim - 1);
    # hand_feature_tmp = gesture_feature(:, global_feature_dim + 1:end);
    # hand_feature_sp = zeros(N_frame, hand_feature_dim * (hand_feature_dim - 1));
    # for k = 1:hand_feature_dim
    #     hand_feature_sp(:, (k - 1) * (hand_feature_dim - 1) + 1:k * (hand_feature_dim - 1)) = hand_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(hand_feature_tmp(:, k), 1, hand_feature_dim - 1);
    # concate all features
    #         fglobal = [gesture_feature(:,1:global_feature_dim), global_feature_sp, gesture_feature_op(:,1:global_feature_dim), gesture_feature_dp1(:,1:global_feature_dim), gesture_feature_dp5(:,1:global_feature_dim), gesture_feature_dp10(:,1:global_feature_dim)];
    #     fh = [gesture_feature(:,global_feature_dim+1:end), hand_feature_sp, gesture_feature_op(:,global_feature_dim+1:end), gesture_feature_dp1(:,global_feature_dim+1:end), gesture_feature_dp5(:,global_feature_dim+1:end), gesture_feature_dp10(:,global_feature_dim+1:end)];
    fglobal = np.concatenate((gesture_feature[:, 0:global_feature_dim], gesture_feature_op[:, 0:global_feature_dim],
                              gesture_feature_dp1[:, 0:global_feature_dim], gesture_feature_dp5[:, 0:global_feature_dim],
                              gesture_feature_dp10[:, 0:global_feature_dim]), axis=1)
    fhand = np.concatenate((gesture_feature[:, global_feature_dim:], gesture_feature_op[:, global_feature_dim:],
                            gesture_feature_dp1[:, global_feature_dim:], gesture_feature_dp5[:, global_feature_dim:],
                            gesture_feature_dp10[:, global_feature_dim:]), axis=1)
    feature_hand = fhand
    feature_global = fglobal
    return feature_hand, feature_global


def extract_all_feature_for_frame_wo_abduction(x, test_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize):
    """
    extract motion feature from a gesture sequence
    """
    # params
    is_filter = 0
    N = x.shape[0]
    J = 22
    global_feature_dim = 6
    hand_feature_dim = 15
    frame_feature_dim = hand_feature_dim + global_feature_dim
    # load hand model parameters
    data_dir = os.path.dirname(__file__)
    if test_id == 2:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_2_187.mat')
    else:
        filename = os.path.join(data_dir, '../../data/natural_hand_joints_1_58.mat')
    dmat = sio.loadmat(filename)
    natural_hand_joints = dmat['natural_hand_joints']
    # filter
    if is_filter:
        x[2:N-2,:] = (-3 * x[0:N-4,:] + 12 * x[1:N-3,:] + 17 * x[2:N-2,:] +
                      12 * x[3:N-1,:] - 3 * x[4:,:]) / 35
    # motion feature
    gesture_feature = np.zeros((N, frame_feature_dim - 1))
    for fid in range(N):  # frame id
        joint = x[fid,:]
        joint = np.reshape(joint, (-1, 3))
        frame_feature = extract_feature_for_frame(joint, natural_hand_joints)
        frame_feature_wo = frame_feature[5:]
        frame_feature_wo = np.reshape(frame_feature_wo, (5, 4))
        frame_feature_wo = frame_feature_wo[:, 1:]
        frame_feature_wo = np.reshape(frame_feature_wo, (1, -1))
        gesture_feature[fid,:] = np.append(frame_feature[0:5], frame_feature_wo)
    # global amp feature
    [global_amp_feature, palm_radius] = extract_feature_global_amp(x, M, max_dist_factor)
    global_amp_feature = np.reshape(global_amp_feature, (-1, 1))
    # combine motion and amp feature
    gesture_feature = np.concatenate((global_amp_feature, gesture_feature), axis=1)
    # offset pose
    gesture_feature_op = gesture_feature - repmat(gesture_feature[0,:], gesture_feature.shape[0], 1)
    # dynamic pose
    gesture_feature_dp1 = gesture_feature
    gesture_feature_dp1[1:,:] = gesture_feature[1:,:] - gesture_feature[0:-1,:]
    of = offset1  # of = 5;
    gesture_feature_dp5 = gesture_feature
    gesture_feature_dp5[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    of = offset2  # of = 10;
    gesture_feature_dp10 = gesture_feature
    gesture_feature_dp10[of:,:] = gesture_feature[of:,:] - gesture_feature[0:-of,:]
    # # static pose
    # global_feature_tmp = gesture_feature[:, 0:global_feature_dim]
    # global_feature_sp = np.zeros(N, global_feature_dim * (global_feature_dim - 1))
    # for k in range(global_feature_dim):
    #     global_feature_sp(:, (k - 1) * (global_feature_dim - 1) + 1:k * (global_feature_dim - 1)) = global_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(global_feature_tmp(:, k), 1, global_feature_dim - 1);
    # hand_feature_tmp = gesture_feature(:, global_feature_dim + 1:end);
    # hand_feature_sp = zeros(N_frame, hand_feature_dim * (hand_feature_dim - 1));
    # for k = 1:hand_feature_dim
    #     hand_feature_sp(:, (k - 1) * (hand_feature_dim - 1) + 1:k * (hand_feature_dim - 1)) = hand_feature_tmp(:, [
    #     1:k - 1, k + 1:end]) - repmat(hand_feature_tmp(:, k), 1, hand_feature_dim - 1);
    # concate all features
    #         fglobal = [gesture_feature(:,1:global_feature_dim), global_feature_sp, gesture_feature_op(:,1:global_feature_dim), gesture_feature_dp1(:,1:global_feature_dim), gesture_feature_dp5(:,1:global_feature_dim), gesture_feature_dp10(:,1:global_feature_dim)];
    #     fh = [gesture_feature(:,global_feature_dim+1:end), hand_feature_sp, gesture_feature_op(:,global_feature_dim+1:end), gesture_feature_dp1(:,global_feature_dim+1:end), gesture_feature_dp5(:,global_feature_dim+1:end), gesture_feature_dp10(:,global_feature_dim+1:end)];
    fglobal = np.concatenate((gesture_feature[:, 0:global_feature_dim], gesture_feature_op[:, 0:global_feature_dim],
                              gesture_feature_dp1[:, 0:global_feature_dim], gesture_feature_dp5[:, 0:global_feature_dim],
                              gesture_feature_dp10[:, 0:global_feature_dim]), axis=1)
    fhand = np.concatenate((gesture_feature[:, global_feature_dim:], gesture_feature_op[:, global_feature_dim:],
                            gesture_feature_dp1[:, global_feature_dim:], gesture_feature_dp5[:, global_feature_dim:],
                            gesture_feature_dp10[:, global_feature_dim:]), axis=1)
    feature_hand = fhand
    feature_global = fglobal
    return feature_hand, feature_global


if __name__ == '__main__':
    print 'motion feature'
