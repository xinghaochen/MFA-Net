% Caculate the parameters of hand model
% Xinghao Chen, 14 Nov, 2016

close all; clear all; clc;

%% dataset directory and other necessary parameters
% dataset_dir = '/media/xiaowei/Work1/Xinghao/Datasets/NYU/';
config;
addpath([dataset_dir, '/']);
data_names = {'train', 'test_1', 'test_2'};
display_image = 0;
split_num = [20, 1, 2];
cube_sizes = [300, 300, 300 * 0.87];
id_starts = [1, 1, 2441];
id_ends = [72757, 2440, 8252];
decimation = 1;
kinect_index = 1;
joint_id = [29,5,3,1,0,11,9,7,6,17,15,13,12,23,21,19,18,28,27,25,24]+1;
% joint_id = 1:36;
J = length(joint_id);
img_size = 96;
label_type = 1;

%% load reference image and joints
D = 3;
id = 2;
id_start = id_starts(D);
id_end = id_ends(D);
chunck_size = (id_end - id_start) / split_num(D);
cube_size = cube_sizes(D);  
data_name = data_names{D};
if strcmp(data_name,'train') 
    data_type = 'train';
else
    data_type = 'test';
end
% load groundtruth joints
if label_type == 0
    load([dataset_dir, '/', data_type, '/joint_data.mat']);
    joint_gt = squeeze(joint_xyz(kinect_index, id,joint_id, :));
else
    load([dataset_dir,'/NYU_21jnt_', data_type, '_ground_truth.mat']);
    % reshape
    joint_xyz = reshape(joint_xyz', 3, 21, [])*1000;
    joint_xyz = permute(joint_xyz,[3, 2, 1]);
    joint_gt = squeeze(joint_xyz(id,:, :));
end
% load an depth image
filename_prefix = sprintf('%d_%07d', kinect_index, id);
depth = imread([dataset_dir, data_type, '/depth_', filename_prefix, '.png']);
depth = uint16(depth(:,:,3)) + bitsll(uint16(depth(:,:,2)), 8);
joint_uvd = convert_xyz_to_uvd_sa(joint_gt);
show_depth_joints(depth, joint_uvd, 1, 1, 1);

%% caculate hand parameters
edges = [1,2;2,3;3,4;4,5;1,6;6,7;7,8;8,9;1,10;10,11;11,12;...
    12,13;1,14;14,15;15,16;16,17;1,18;18,19;19,20;20,21];
bone_vec = joint_gt(edges(:,1),:) - joint_gt(edges(:,2),:);
bone_lengths = sqrt(sum(bone_vec.^2, 2));
natural_hand_joints = joint_gt - repmat(joint_gt(1,:) ,J ,1);
save('results/hand_parameters_new.mat', 'bone_lengths', 'natural_hand_joints');

figure;
show_3d_joints(joint_gt, 1, 1, 1);