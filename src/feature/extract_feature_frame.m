function feature = extract_feature_frame(joint, natural_hand_joints)
% Extract skelton feature from joint
% including global translation, global rotation and bone angles
%   Xinghao Chen, 20 Dec, 2016

    [bone_lengths, ~] = calculate_hand_parameters(joint);
    % normalize palm bone
    for id = [1,7,11,15,19]
        b1 = norm(joint(id,:) - joint(2,:));
        b2 = norm(natural_hand_joints(id,:) - natural_hand_joints(2,:));
        natural_hand_joints(id,:) = natural_hand_joints(2,:) + b1/b2*(natural_hand_joints(id,:) - natural_hand_joints(2,:));
    end
    b1 = norm(joint(3,:) - joint(1,:));
    b2 = norm(natural_hand_joints(3,:) - natural_hand_joints(1,:));
    natural_hand_joints(3,:) = natural_hand_joints(1,:) + b1/b2*(natural_hand_joints(3,:) - natural_hand_joints(1,:));
    
    [bone_angle, global_tral, global_rot] = inverse_kinematic(joint, bone_lengths, natural_hand_joints);
    [a, e, r] = cart2sph(global_tral(1),global_tral(2),global_tral(3));
    eul = tform2eul(global_rot);
    % normalize
    bone_angle(:,2:end) = bone_angle(:,2:end) + pi/4;
    feature_global = [a, e, eul];
    feature_hand = reshape(bone_angle, 1, []);
    feature = [feature_global, feature_hand];

end

