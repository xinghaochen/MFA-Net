close all; clear all; clc;

test_id = 2;
seq_id = 1;
do_pca = 0;
pca_dim = 24;
is_full = 1;
M = 5;
max_dist_factor = 1.5;
offset1 = 5;
offset2 = 10;
is_normalize = 1;
[feature_hand, feature_global, y] = extract_features_for_alldata(is_full, test_id, seq_id, do_pca, pca_dim,  M, max_dist_factor, offset1, offset2, is_normalize);
