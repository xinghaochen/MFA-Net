function ShowSequence(seq, label, is_show_connect, is_show_id)
% visualize a gesture sequence
% seq: N*22*3
%   Xinghao Chen, 3 Jan, 2017

    lines = [0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20;
             2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]+1;
    N_frame = size(seq, 1);
    seq = reshape(seq, N_frame, 3, []);
    seq = permute(seq, [1,3,2]);
    J = size(seq, 2);
    xmax = max(max(seq(:, :, 1)));
    xmin = min(min(seq(:, :, 1)));
    ymax = max(max(seq(:, :, 2)));
    ymin = min(min(seq(:, :, 2)));
    zmax = max(max(seq(:, :, 3)));
    zmin = min(min(seq(:, :, 3)));
    xmax, xmin
    ymax, ymin
    zmax, zmin
    for fid = 1:9:N_frame
%         clf;
        hold on;
        scatter3(seq(fid, :, 1), seq(fid, :, 2), seq(fid, :, 3), 'b.');
        xlim([xmin, xmax]);
        ylim([ymin, ymax]);
        zlim([zmin, zmax]);
        xlabel('X axis');
        ylabel('Y axis');
        zlabel('Z axis');
        if is_show_id
            hold on;
            for i = 1:J
                text(seq(fid, :, 1), seq(fid, :, 2), seq(fid, :, 3), num2str(i));
            end
        end
        if is_show_connect
            hold on;
            for i = 1:size(lines, 2)
                id1 = lines(1, i);
                id2 = lines(2, i);
                plot3([seq(fid, id1, 1), seq(fid,  id2, 1)],...
                     [seq(fid,  id1, 2), seq(fid,  id2, 2)],...
                     [seq(fid,  id1, 3), seq(fid,  id2, 3)], 'r');
            end
        end
        title(num2str(label));
%         view([90 0]);
        view([70,30]);
        set(gcf,'Position',[30 20 900 900]);
        grid off;
%         saveas(gcf,['testid_1_train_141/skeleton_',num2str(fid)],'epsc');
        pause(0.1);
end

