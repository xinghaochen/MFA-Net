function [bone_angle, global_tral, global_rot] = inverse_kinematic(joints, bone_lengths, natural_hand_joints)
% Inverse kinematic algorithm to convert the joint locations to the angle of the bones 
% bone_angle: 20-D
% bone_length: 20-D
% global_tral: global position, 3-D
% global_rot: global orientation, 3-D
%   Xinghao Chen, 25 Nov, 2016

    %% parameters
    J = size(natural_hand_joints, 1);
    wrist_id = 1;
    palm_id = 2;
    mcp_id = [3:4:19];
    pip_id = [4:4:20];
    dip_id = [5:4:21];
    tip_id = [6:4:22];
    finger_id = [mcp_id;pip_id;dip_id;tip_id];
    mcp_bone_id = [1:4:17];
    pip_bone_id = [2:4:18];
    dip_bone_id = [3:4:19];
    tip_bone_id = [4:4:20];
    
    %% global translation and rotation
    [global_rot_tmp, global_tral, ~] = Kabsch(natural_hand_joints([wrist_id,mcp_id],:)', joints([wrist_id,mcp_id],:)');
    global_rot = eye(4);
    global_rot(1:3,1:3) = global_rot_tmp;
    
    %% angle of bones
    bone_angle = zeros(5, 4);
    % first inverse global translation and rotation
    joint_hg = ones(4, J);
    joint_hg(1:3, :) = joints';
    joint_hg = inv(global_rot) * (joint_hg - repmat([global_tral;0], 1, J));
    
%     figure;
%     joint_t = joint_hg';
%     joint_t = joint_t(:,1:3);
%     show_3d_joints(joint_t, 1, 1, 1);
%     pause
        
    for idx = 1:5
        % mcp-pip z rotation
        vec = joint_hg(1:2, dip_id(idx)) - joint_hg(1:2, mcp_id(idx));
        theta = atan(vec(1)/vec(2));
        if abs(theta) > pi/2
            theta = sign(theta)*abs(abs(theta) - pi);
        end
%                 if theta > 0
%             theta = theta - pi;
%         end
        theta = -theta;
        bone_angle(idx,1) = theta;
        % reverse translation and rotation
        Tf = makehgtform('translate', -joint_hg(1:3, mcp_id(idx)));
        Rf = makehgtform('zrotate', -theta);
        finger = joint_hg(:, finger_id(1:4,idx));
        finger =  Rf * Tf * finger;
        joint_hg(:, finger_id(1:4,idx)) = finger;
        
%         figure;
%         joint_t = joint_hg';
%         joint_t = joint_t(:,1:3);
%         show_3d_joints(joint_t, 1, 1, 1);
%         pause
        
        % mcp-pip x rotation
        vec = joint_hg(2:3, pip_id(idx)) - joint_hg(2:3, mcp_id(idx));
        theta = atan(vec(2)/vec(1));
%         if theta < -pi/4
%             theta = theta + pi;
%         end
        if theta > pi/8
            theta = theta - pi;
        end
        bone_angle(idx,2) = theta;
        % reverse translation and rotation
        Tf = makehgtform('translate', -joint_hg(1:3, pip_id(idx)));
        Rf = makehgtform('xrotate', -theta);
        finger = joint_hg(:, finger_id(2:4,idx));
        finger = Rf * Tf * finger;
        joint_hg(:, finger_id(2:4,idx)) = finger;
        
%         figure;
%         joint_t = joint_hg';
%         joint_t = joint_t(:,1:3);
%         show_3d_joints(joint_t, 1, 1, 1);
%         pause

        % pip-dip x rotation
        vec = joint_hg(2:3, dip_id(idx)) - joint_hg(2:3, pip_id(idx));
        theta = atan(vec(2)/vec(1));
        if theta > pi/8
            theta = theta - pi;
        end
        bone_angle(idx,3) = theta;
        % reverse translation and rotation
        Tf = makehgtform('translate', -joint_hg(1:3, dip_id(idx)));
        Rf = makehgtform('xrotate', -theta);
        finger = joint_hg(:, finger_id(3:4,idx));
        finger = Rf * Tf * finger;
        joint_hg(:, finger_id(3:4,idx)) = finger;

        % dip-tip x rotation
        vec = joint_hg(2:3, tip_id(idx)) - joint_hg(2:3, dip_id(idx));
        theta = atan(vec(2)/vec(1));
        if theta > pi/8
            theta = theta - pi;
        end
        bone_angle(idx,4) = theta;
    end

end

