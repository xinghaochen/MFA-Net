% Extract features from dynamic hand gesture squences
% Xinghao Chen, 18 Dec, 2016

close all; clear all; clc;

%% load data
load('../../data/DHGdata/DHGdata.mat');

N_train = length(x_train);
N_test = length(x_test);
J = 22;
is_show = 1;
frame_feature_dim = 25;
type_names = {'train', 'test'};
type_id= 1;

%% ref id to calculate the global rotation
refid = 58;
joint = x_train{1,refid}(1,:);
joint = reshape(joint, 3, [])';
[~, natural_hand_joints] = calculate_hand_parameters(joint);
% show_3d_joints(natural_hand_joints, 1, 1, 1);
% pause;

%% extract per frame hand parameters
if type_id == 1
    N = N_train;
    x = x_train;
    y = y_train;
else
    N = N_test;
    x = x_test;
    y = y_test;    
end
feature = cell(1, N);

for sid = 1:N % sample id
    disp(sid)
    sid = ceil(rand(1)*N)
    sid = 2660;
    N_frame= size(x{1,sid},1);
    for fid = 1: 1%N_frame% frame id
        joint = x{1,sid}(fid,:);
        joint = reshape(joint, 3, [])';
        
        [bone_lengths, ~] = calculate_hand_parameters(joint);
        [bone_angle, global_tral, global_rot] = inverse_kinematic(joint, bone_lengths, natural_hand_joints);
        [a, b, c] = cart2sph(global_tral(1),global_tral(2),global_tral(3))
        tform2eul(global_rot)
         
        % display 3d joints
        if is_show
            close all;
            show_3d_joints(joint, 1, 1, 1);
            % view([90 0]);
            figure;
            joint_fk = forward_kinematic(bone_angle, bone_lengths, global_tral, global_rot, natural_hand_joints);
            show_3d_joints(joint_fk, 1, 1, 1);
            pause;
        end
    end
end
%% extract per frame hand parameters
if type_id == 1
    save('../data/feature_train.mat', 'feature', 'y_train');
else
    save('../data/feature_test.mat', 'feature', 'y_test');
end