% Extract features from dynamic hand gesture squences
% Xinghao Chen, 18 Dec, 2016

close all; clear all; clc;

%% load data
% load('../../data/DHGdata/DHGdata_testid_1.mat');
load('../../data/DHGdata/DHGdata_testid_2.mat');

N_train = length(x_train);
N_test = length(x_test);
J = 22;
is_show = 0;
frame_feature_dim = 25;
type_names = {'train', 'test'};
type_id= 1;
save_result = 1;
coarse_fine_binary = 0;
coarse_fine_split = 0;
do_pca = 0;
pca_dim = 24;


%% ref id to calculate the global rotation
% refid = 58;
refid = 187;
% for refid=187:1000
joint = x_train{1,refid}(1,:);
joint = reshape(joint, 3, [])';
[~, natural_hand_joints] = calculate_hand_parameters(joint);
hold off;
show_3d_joints(natural_hand_joints, 1, 1, 1);
% save('natural_hand_joints_1_58.mat', 'natural_hand_joints');
save('natural_hand_joints_2_187.mat', 'natural_hand_joints');
view([90,0]);
% pause;
% end
return


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
feature_array = [];

for sid = 1:N % sample id
    disp(sid)
    N_frame= size(x{1,sid},1);
    gesture_feature = zeros(N_frame, frame_feature_dim);
    for fid = 1: N_frame% frame id
        joint = x{1,sid}(fid,:);
        joint = reshape(joint, 3, [])';
        
        hand_feature = extract_feature_frame(joint, natural_hand_joints);
        gesture_feature(fid, :) = hand_feature;
    end
    % offset pose
    gesture_feature_op = gesture_feature - repmat(gesture_feature(1,:), size(gesture_feature,1), 1);
    % dynamic pose
    gesture_feature_dp1 = zeros(N_frame, frame_feature_dim);
    gesture_feature_dp1(1+1:end,:) = gesture_feature(1+1:end,:) - gesture_feature(1:end-1,:);
    gesture_feature_dp5 = zeros(N_frame, frame_feature_dim);
    gesture_feature_dp5(1+5:end,:) = gesture_feature(1+5:end,:) - gesture_feature(1:end-5,:);
    gesture_feature_dp10 = zeros(N_frame, frame_feature_dim);
    gesture_feature_dp10(1+10:end,:) = gesture_feature(1+10:end,:) - gesture_feature(1:end-10,:);
    % concate all features
    fglobal = [gesture_feature(:,1:5), gesture_feature_op(:,1:5), gesture_feature_dp1(:,1:5), gesture_feature_dp5(:,1:5), gesture_feature_dp10(:,1:5)];
    fh = [gesture_feature(:,6:end), gesture_feature_op(:,6:end), gesture_feature_dp1(:,6:end), gesture_feature_dp5(:,6:end), gesture_feature_dp10(:,6:end)];
%     gesture_feature_final = [fglobal, fh];
    gesture_feature_final = fh;

    feature(sid) = {gesture_feature_final};
    feature_array = [feature_array;gesture_feature_final];
end

%% coarse or fine binary label
if coarse_fine_binary
    fine_class = [1,3:6]-1;
    coarse_class = [2,7:14]-1;
    for c = 0:13
        idx = find(y_train == c);
        if ~isempty(find(fine_class == c, 1))
            y_train(idx) = 0;
        else
            y_train(idx) = 1;
        end
        idx = find(y_test == c);
        if ~isempty(find(fine_class == c, 1))
            y_test(idx) = 0;
        else
            y_test(idx) = 1;
        end
    end
end

if coarse_fine_split == 1 % coarse
    fine_class = [1,3:6]-1;
    coarse_class = [2,7:14]-1;
    y_train_new = y_train;
    y_test_new = y_test;
    for c = 0:13
        idx = find(y_train_new == c);
        if ~isempty(find(fine_class == c, 1))
            y_train(idx) = 9;
        else
            y_train(idx) = find(coarse_class == c)-1;
        end
        idx = find(y_test_new == c);
        if ~isempty(find(fine_class == c, 1))
            y_test(idx) = 9;
        else
            y_test(idx) = find(coarse_class == c)-1;
        end
    end
elseif coarse_fine_split == 2 % fine
    fine_class = [1,3:6]-1;
    coarse_class = [2,7:14]-1;
    y_train_new = y_train;
    y_test_new = y_test;
    for c = 0:13
        idx = find(y_train_new == c);
        if ~isempty(find(coarse_class == c, 1))
            y_train(idx) = 5;
        else
            y_train(idx) = find(fine_class == c)-1;
        end
        idx = find(y_test_new == c);
        if ~isempty(find(coarse_class == c, 1))
            y_test(idx) = 5;
        else
            y_test(idx) = find(fine_class == c)-1;
        end
    end
end

%% save result
% if save_result
%     if type_id == 1
%         save('../data/feature_hand_fine_train.mat', 'feature', 'y_train');
%     else
%         save('../data/feature_hand_fine_test.mat', 'feature', 'y_test');
%     end
% end
    
%% pca
if do_pca
    if type_id == 1
        [coeff,score,latent,tsquared,explained,mu]= pca(feature_array(:,26:end));
        save('../data/pca_hand_params.mat', 'coeff', 'score', 'latent', 'mu', 'tsquared', 'explained');
        [coeff_g,score_g,latent_g,tsquared_g,explained_g,mu_g]= pca(feature_array(:,1:25));
        save('../data/pca_global_params.mat', 'coeff_g', 'score_g', 'latent_g', 'mu_g', 'tsquared_g', 'explained_g');
    else
        load('../data/pca_hand_params.mat');
        load('../data/pca_global_params.mat');
   end
    feature_pca = cell(1, N);
    for sid = 1:N
        N_frame= size(x{1,sid},1);
        gesture_feature_pca = zeros(N_frame, pca_dim);
        for fid = 1: N_frame% frame id
            gesture_feature_pca_global = (feature{sid}(fid,1:25) - mu_g) * coeff_g(:,1:12);
            gesture_feature_pca_hand = (feature{sid}(fid,26:end) -mu) * coeff(:,1:pca_dim-12);
            gesture_feature_pca(fid,:) = [gesture_feature_pca_global,gesture_feature_pca_hand];
        end
        feature_pca(sid) = {gesture_feature_pca};
    end

    % save result
    if save_result
        if type_id == 1
            save('../data/feature_pca_train.mat', 'feature_pca', 'y_train');
        else
            save('../data/feature_pca_test.mat', 'feature_pca', 'y_test');
        end
    end
end