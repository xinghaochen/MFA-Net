function [joint] = forward_kinematic(bone_angle, bone_lengths, global_tral, global_rot, natural_hand_joints)
% Forward kinematic algorithm to convert the angle of the bones to the joint locations 
% bone_angle: 20-D
% bone_length: 20-D
% global_tral: global position, 3-D
% global_rot: global orientation, 3-D
%   Xinghao Chen, 1 Nov, 2016
    
    %% parameters
    J = size(natural_hand_joints, 1);
    wrist_id = 1;
    palm_id = 2;
    mcp_id = [3:4:19];
    pip_id = [4:4:20];
    dip_id = [5:4:21];
    tip_id = [6:4:22];
    mcp_bone_id = [1:4:17];
    pip_bone_id = [2:4:18];
    dip_bone_id = [3:4:19];
    tip_bone_id = [4:4:20];
    joint_hg = ones(4, J);
    
    %% the wrist joints and 5 MCP joints are fixed.
    joint_hg(1:3, wrist_id) = natural_hand_joints(wrist_id, :)';
    joint_hg(1:3, palm_id) = natural_hand_joints(palm_id, :)';
    joint_hg(1:3, mcp_id) = natural_hand_joints(mcp_id, :)';
    
    %% adjust the PIP, DIP, TIP according to the bone_angle
    for idx = 1:5
        % local coordinates
        finger = zeros(4, 3);
        finger(4,:) = 1;
        % tip
        Tpip = makehgtform('translate', [0, bone_lengths(tip_bone_id(idx)), 0]);
        Rpip = makehgtform('xrotate', sum(bone_angle(idx, 4:4)));
        finger(:, 3:end) = Rpip * Tpip * finger(:, 3:end);
        % dip
        Tpip = makehgtform('translate', [0, bone_lengths(dip_bone_id(idx)), 0]);
        Rpip = makehgtform('xrotate', sum(bone_angle(idx, 3:3)));
        finger(:, 2:end) = Rpip * Tpip * finger(:, 2:end);        
        % pip
        Tpip = makehgtform('translate', [0, bone_lengths(pip_bone_id(idx)), 0]);
        Rpip = makehgtform('xrotate', bone_angle(idx, 2));
        finger = Rpip * Tpip * finger;
        
        % transform to global coordinates
        Tf = makehgtform('translate', joint_hg(1:3, mcp_id(idx)));
        Rf = makehgtform('zrotate', bone_angle(idx, 1));
        
        
%         if idx ~= 5
            finger = Tf * Rf * finger;
%         else
%             finger(:,2:end) = Rf * finger(:,2:end);
%             finger = Tf * finger;
%         end
        joint_hg(:, [4:6]+(idx-1)*4) = finger;
    end
    
    %% global translation
%     Rgx = makehgtform('xrotate', global_rot(1));
%     Rgy = makehgtform('yrotate', global_rot(2));
%     Rgz = makehgtform('zrotate', global_rot(3));
%     joint_hg = Rgz * Rgy * Rgx * joint_hg;
%     % global translation and rotation
%     Tgtran = makehgtform('translate', global_pos);
%     joint_hg = Tgtran * joint_hg;
        
    joint_hg = global_rot * joint_hg + repmat([global_tral;0], 1, J);

    % return
    joint_hg = joint_hg';
    joint = joint_hg(:,1:3);
end

