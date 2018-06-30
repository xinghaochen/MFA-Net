function show_3d_joints(joint, flag_xyz, show_connection, show_id )
% display the joints in 3D
%   Xinghao Chen, 15 Nov, 2016

    J = size(joint, 1);
    % convert to uvd
    if flag_xyz == 1
        jnt_xyz = joint;
    else
        jnt_xyz = convert_uvds_to_xyzs(reshape(joint, [1, J, 3]));
        jnt_xyz = squeeze(jnt_xyz);
    end
    % display joints
    jnt_colors = rand(J, 3);
    scatter3(jnt_xyz(:,1), jnt_xyz(:,2), jnt_xyz(:,3), 40, jnt_colors, 'filled');
    if show_connection
%         edges = [1,2;2,3;3,4;4,5;1,6;6,7;7,8;8,9;1,10;10,11;11,12;...
%             12,13;1,14;14,15;15,16;16,17;1,18;18,19;19,20;20,21];
        edges = [0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20;
         2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]+1;
        edges = edges';
        for j = 1:size(edges, 1)
            hold on
            line(jnt_xyz([edges(j,1),edges(j,2)],1), jnt_xyz([edges(j,1),edges(j,2)],2), jnt_xyz([edges(j,1),edges(j,2)],3),'LineWidth',1, 'color', 'b');
        end
    end
    if show_id
        for j = 1:J
            hold on
            text(jnt_xyz(j,1), jnt_xyz(j,2), jnt_xyz(j,3), num2str(j), 'FontSize', 15, 'FontName', 'Ubuntu');
        end
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
end

