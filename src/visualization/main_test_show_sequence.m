close all; clear all; clc;

is_show_connect = 1;
is_show_id = 0;
is_full = 0;
test_id = 1;
fid = 141;

%% load data
if is_full
    load(['../data/DHGdata/DHGdata_full_testid_', num2str(test_id), '.mat']);
else
    load(['../data/DHGdata/DHGdata_testid_', num2str(test_id), '.mat']);
end

%% show
seq = x_train{1, fid};
label = y_train(fid);
ShowSequence(seq, label, is_show_connect, is_show_id);
