function [bone_lengths, natural_hand_joints] = calculate_hand_parameters(joint)
% Caculate the parameters of hand model
% Xinghao Chen, 18 Dec, 2016

% J = 22;
% edges = [0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20;
%          2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]+1;
% edges = edges';
J = 22;
joint21 = joint([1,3:end], :);
edges = [1,2;2,3;3,4;4,5;1,6;6,7;7,8;8,9;1,10;10,11;11,12;...
    12,13;1,14;14,15;15,16;16,17;1,18;18,19;19,20;20,21];
bone_vec = joint21(edges(:,1),:) - joint21(edges(:,2),:);
bone_lengths = sqrt(sum(bone_vec.^2, 2));
natural_hand_joints = joint - repmat(joint(1,:) ,J ,1);

end