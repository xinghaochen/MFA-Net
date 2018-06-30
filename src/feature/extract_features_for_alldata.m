function [feature_hand, feature_global, y] = extract_features_for_alldata(is_full, test_id, seq_id, do_pca, pca_dim, M, max_dist_factor, offset1, offset2, is_normalize)
% Extract features from dynamic hand gesture squences for all samples in
% the dataset
% test_id: the id of subject which is used for testing
% seq_id: 0 - train,  1 - test
% Xinghao Chen, 23 Dec, 2016

%folder = '/home/icvl/xinghao/robotics';
%addpath(genpath(folder));

M = double(M);
offset1 = double(offset1);
offset2 = double(offset2);

%% load data
if is_full
    load(['data/DHGdata/DHGdata_full_testid_', num2str(test_id), '.mat']);
else
    load(['data/DHGdata/DHGdata_testid_', num2str(test_id), '.mat']);
end

N_train = length(x_train);
N_test = length(x_test);
J = 22;
global_feature_dim = 6;
hand_feature_dim = 20;
frame_feature_dim = hand_feature_dim+global_feature_dim;
type_names = {'train', 'test'};
save_result = 0;
coarse_fine_binary = 0;
coarse_fine_split = 0;
% do_pca = 1;
% pca_dim = 24;
is_filter = 0;

%% ref id to calculate the global rotation
% refid = 58;
% joint = x_train{1,refid}(1,:);
% joint = reshape(joint, 3, [])';
% [~, natural_hand_joints] = calculate_hand_parameters(joint);
if test_id == 2
    load('natural_hand_joints_2_187.mat');
else
    load('natural_hand_joints_1_58.mat');
end

%% extract per frame hand parameters
if seq_id == 0
    N = N_train;
    x = x_train;
    y = y_train;
else
    N = N_test;
    x = x_test;
    y = y_test;
end

feature_hand = cell(1, N);
feature_global = cell(1, N);

tic
for sid = 1:N % sample id
    if mod(sid, 500) == 0
        disp(sid)
    end
    % filter
    if is_filter
       x{1,sid}(3:end-2, :) = (-3*x{1,sid}(1:end-4, :)+12*x{1,sid}(2:end-3, :)+17*x{1,sid}(3:end-2, :)...
           +12*x{1,sid}(4:end-1, :)-3*x{1,sid}(5:end, :)) / 35;
    end
    N_frame= size(x{1,sid},1);
    gesture_feature = zeros(N_frame, frame_feature_dim-1);
    for fid = 1: N_frame% frame id
        joint = x{1,sid}(fid,:);
        joint = reshape(joint, 3, [])';
        
        frame_feature = extract_feature_frame(joint, natural_hand_joints);
        gesture_feature(fid, :) = frame_feature;
    end
    [global_amp_feature,palm_radius] = extract_feature_global_amp(x{1,sid}, M, max_dist_factor);
    gesture_feature = [global_amp_feature, gesture_feature];
    % offset pose
    gesture_feature_op = gesture_feature - repmat(gesture_feature(1,:), size(gesture_feature,1), 1);
    % dynamic pose
    gesture_feature_dp1 = gesture_feature;
    gesture_feature_dp1(1+1:end,:) = gesture_feature(1+1:end,:) - gesture_feature(1:end-1,:);
    of = offset1;%of = 5;
    gesture_feature_dp5 = gesture_feature;
    gesture_feature_dp5(1+of:end,:) = gesture_feature(1+of:end,:) - gesture_feature(1:end-of,:);
    of = offset2;%of = 10;
    gesture_feature_dp10 = gesture_feature;
    gesture_feature_dp10(1+of:end,:) = gesture_feature(1+of:end,:) - gesture_feature(1:end-of,:);
    % static pose
    global_feature_tmp = gesture_feature(:,1:global_feature_dim);
    global_feature_sp = zeros(N_frame, global_feature_dim*(global_feature_dim-1));
    for k = 1:global_feature_dim
        global_feature_sp(:,(k-1)*(global_feature_dim-1)+1:k*(global_feature_dim-1)) = global_feature_tmp(:,[1:k-1,k+1:end]) - repmat(global_feature_tmp(:,k), 1, global_feature_dim-1);
    end
    hand_feature_tmp = gesture_feature(:,global_feature_dim+1:end);
    hand_feature_sp = zeros(N_frame, hand_feature_dim*(hand_feature_dim-1));
    for k = 1:hand_feature_dim
        hand_feature_sp(:,(k-1)*(hand_feature_dim-1)+1:k*(hand_feature_dim-1)) = hand_feature_tmp(:,[1:k-1,k+1:end]) - repmat(hand_feature_tmp(:,k), 1, hand_feature_dim-1);
    end
    % concate all features
%         fglobal = [gesture_feature(:,1:global_feature_dim), global_feature_sp, gesture_feature_op(:,1:global_feature_dim), gesture_feature_dp1(:,1:global_feature_dim), gesture_feature_dp5(:,1:global_feature_dim), gesture_feature_dp10(:,1:global_feature_dim)];
%     fh = [gesture_feature(:,global_feature_dim+1:end), hand_feature_sp, gesture_feature_op(:,global_feature_dim+1:end), gesture_feature_dp1(:,global_feature_dim+1:end), gesture_feature_dp5(:,global_feature_dim+1:end), gesture_feature_dp10(:,global_feature_dim+1:end)];
    fglobal = [gesture_feature(:,1:global_feature_dim), gesture_feature_op(:,1:global_feature_dim), gesture_feature_dp1(:,1:global_feature_dim), gesture_feature_dp5(:,1:global_feature_dim), gesture_feature_dp10(:,1:global_feature_dim)];
    fh = [gesture_feature(:,global_feature_dim+1:end), gesture_feature_op(:,global_feature_dim+1:end), gesture_feature_dp1(:,global_feature_dim+1:end), gesture_feature_dp5(:,global_feature_dim+1:end), gesture_feature_dp10(:,global_feature_dim+1:end)];
%     gesture_feature_final = [fglobal, fh];
    gesture_feature_final = fh;

    feature_hand(sid) = {fh};
    feature_global(sid) = {fglobal};

end

if is_normalize
    % normalize
    feature_hand_array = cell2mat(feature_hand');
    feature_global_array = cell2mat(feature_global');
    feature_hand_mean = mean(feature_hand_array);
    feature_hand_std = std(feature_hand_array);
    feature_global_mean = mean(feature_global_array);
    feature_global_std = std(feature_global_array);

    for sid = 1:N % sample id
        if mod(sid, 500) == 0
            disp(sid)
        end
        N_frame= size(x{1,sid},1);
        for fid = 1: N_frame% frame id
            feature_hand{1,sid}(fid, :) = (feature_hand{1,sid}(fid, :) - feature_hand_mean) ./ feature_hand_std;
            feature_global{1,sid}(fid, :) = (feature_global{1,sid}(fid, :) - feature_global_mean) ./ feature_global_std;
        end
    end
end
toc

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
if save_result
    if seq_id == 1
        save('../data/feature_hand_global_train.mat', 'feature_hand', 'feature_global', 'y_train');
    else
        save('../data/feature_hand_global_test.mat', 'feature_hand', 'feature_global', 'y_test');
    end
end
    
%% pca
tic
if do_pca
    % caculate pca parameters
    feature_array = [];
    feature_array = cell2mat(feature_hand');
%     for sid = 1:N 
%         feature_array = [feature_array;feature_hand];
%     end
    if seq_id == 0
        [coeff,score,latent,tsquared,explained,mu]= pca(feature_array);
        save('data/pca_hand_params.mat', 'coeff', 'score', 'latent', 'mu', 'tsquared', 'explained');
%         [coeff_g,score_g,latent_g,tsquared_g,explained_g,mu_g]= pca(feature_array(:,1:25));
%         save('../data/pca_global_params.mat', 'coeff_g', 'score_g', 'latent_g', 'mu_g', 'tsquared_g', 'explained_g');
    else
        load('data/pca_hand_params.mat');
%         load('../data/pca_global_params.mat');
    end
    % apply pca and caculate normalized parameters
    feature_array_pca = (feature_array - repmat(mu, size(feature_array,1),1)) * coeff(:,1:pca_dim);
    feature_array_pca_mean = mean(feature_array_pca);
    feature_array_pca_std = std(feature_array_pca);
    % get final feature
    feature_hand_pca_nor = cell(1, N);
    for sid = 1:N
        N_frame= size(x{1,sid},1);
        gesture_feature_pca = zeros(N_frame, pca_dim);
        for fid = 1: N_frame% frame id
            gesture_feature_pca_hand = (feature_hand{sid}(fid,:) -mu) * coeff(:,1:pca_dim);
            gesture_feature_pca_hand = (gesture_feature_pca_hand - feature_array_pca_mean) ./ feature_array_pca_std;
            gesture_feature_pca(fid,:) = [gesture_feature_pca_hand];
        end
        feature_hand_pca_nor(sid) = {gesture_feature_pca};
    end
    feature_hand = feature_hand_pca_nor;

    % save result
    if save_result
        if seq_id == 1
            save('data/feature_pca_train.mat', 'feature_pca', 'y_train');
        else
            save('data/feature_pca_test.mat', 'feature_pca', 'y_test');
        end
    end
end
toc

%% end of function
end
