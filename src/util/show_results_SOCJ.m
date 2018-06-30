close all; clear all; clc;

test_subjects_num = 20;
is_full = 1;
class_num = 14*(is_full+1);
cal_larfd = 1;
folder = '../../result/DHG_SoCJ/';
prefix = 'result_14_DHGdataset_SOCJ.txt';

%% larfd metric
if cal_larfd
    load([folder,'result_28_DHGdataset_SOCJ.mat']);
    Mgf = confusion_matrix_all;
    load([folder,'result_14_DHGdataset_SOCJ.mat']);
    Mg = confusion_matrix_all;
    larfd = larfd_metric(Mg, Mgf);
    larfd_mean = mean(larfd);
    return; 
end

%% analysis all results
confusion_matrix = zeros(class_num,class_num,test_subjects_num);
recognition_rate = zeros(test_subjects_num, 1);
recognition_rate_fine = zeros(test_subjects_num, 1);
recognition_rate_coarse = zeros(test_subjects_num, 1);

% load data
fid = fopen([folder,'/result_', num2str(class_num), '_DHGdataset_SOCJ.txt']);
C = textscan(fid, '%d%d%d%d%d');

for test_id = 1:test_subjects_num
    idx = (C{1,3} == test_id);
    if is_full
        y_test = C{1,1}(idx)*2+C{1,2}(idx)-2;
    else
        y_test = C{1,1}(idx);
    end
    pred_label = C{1,5}(idx);

    N_test = length(y_test);
    class_samples_num = ones(class_num, 1)*(10/(is_full+1));

    % calulate confusion matrix
    confusion_matrix(:,:,test_id) = confusionmat(y_test, pred_label) ./ repmat(class_samples_num, 1, class_num);
    recognition_rate(test_id,:) = mean(diag(confusion_matrix(:,:,test_id)));

    t = diag(confusion_matrix(:,:,test_id));
    
    if is_full
        fine_class = [1,2,5:12];
        coarse_class = [3,4,13:28];       
    else
        fine_class = [1,3:6];
        coarse_class = [2,7:14];
    end
    recognition_rate_fine(test_id,:) = mean(t(fine_class));
    recognition_rate_coarse(test_id,:) = mean(t(coarse_class));
end

%% draw results
rate = [recognition_rate,recognition_rate_fine,recognition_rate_coarse];
confusion_matrix_all = mean(confusion_matrix, 3);
% recognition_rate_all = mean(recognition_rate)
% recognition_rate_fine_all = mean(recognition_rate_fine)
% recognition_rate_coarse_all = mean(recognition_rate_coarse)
recognition_rate_max = max(rate)
recognition_rate_min = min(rate)
recognition_rate_mean = mean(rate)
recognition_rate_std = std(rate)
if is_full
    tick = {'G(1)';'G(2)';'T(1)';'T(2)';'E(1)';'E(2)';'P(1)';'P(2)';'R-CW(1)';'R-CW(2)';
        'R-CCW(1)';'R-CCW(2)';'S-R(1)';'S-R(2)';'S-L(1)';'S-L(2)';'S-U(1)';'S-U(2)';
        'S-D(1)';'S-D(2)';'S-X(1)';'S-X(2)';'S-V(1)';'S-V(2)';'S-+(1)';'S-+(2)';'Sh(1)';'Sh(2)'};
else
    tick = {'G';'T';'E';'P';'R-CW';'R-CCW';'S-R';'S-L';'S-U';'S-D';'S-X';'S-V';'S-+';'Sh'};
end
figure('Position',[300 200 1600 900]);
draw_cm(confusion_matrix_all*100, tick, class_num);
set(gca,'Units','normalized','Position',[0.05 0.08 0.93 0.9]);  %# Modify axes size

save([folder,'/result_', num2str(class_num), '_DHGdataset_SOCJ.mat'], 'rate', 'confusion_matrix_all');

% saveas(gcf,'cm_feature_hand_global_skeleton_full','epsc');
% saveas(gcf,'cm_feature_hand_global_skeleton_full.bmp');