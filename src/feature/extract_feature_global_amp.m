function [global_amp_feature,palm_radius] = extract_feature_global_amp(seq, M, max_dist_factor)
% Extract global translation ampitude feature from a gesture sequence
%   Xinghao Chen,  27 Dec, 2016

%     M = 5;
%     max_dist_factor = 1.5;
    
    palm_id = 2;
    palm_xyz_id = 4:6;
    wrist_xyz_id = 1:3;
    mid_mcp_xyz_id = 31:33;
    
    N = size(seq, 1);
    global_amp_feature = seq(:, palm_xyz_id) - repmat(seq(1, palm_xyz_id), N, 1);
    global_amp_feature = sqrt(sum(global_amp_feature.^2, 2));
    palm_radius = sqrt(sum( (seq(:, palm_xyz_id) - seq(:, wrist_xyz_id)).^2, 2)) + ...
                  sqrt(sum( (seq(:, palm_xyz_id) - seq(:, mid_mcp_xyz_id)).^2, 2));
    palm_radius = mean(palm_radius);
    
    % bin
    thres = zeros(1, M);
    sigma = palm_radius;
    mu = 0;
%     max_range_prob = normcdf(-sigma*max_dist_factor, mu, sigma);
%     for k = 1:M
%         prob = (1-(1-2*max_range_prob)/(M-k+1))/2;
%         thres(k) = -norminv(prob,mu,sigma);
%     end
    max_range_prob = normcdf(-sigma*max_dist_factor, mu, sigma);
    for k = 1:M
        prob = (1-(1-2*max_range_prob)/(M-k+1))/2;
        thres(k) = -norminv(prob,mu,sigma);
    end
    thres = [0,thres];
    % thresholding
    global_amp_feature_bin = zeros(size(global_amp_feature));
    for k = 2:M+1
        idx = find(global_amp_feature >= thres(k-1) & global_amp_feature < thres(k));
        global_amp_feature_bin(idx) = k-1;
    end
    idx = find(global_amp_feature > thres(M));
    global_amp_feature_bin(idx) = M+1;
    global_amp_feature_bin = (global_amp_feature_bin-1)*1.0 ./ M;
    global_amp_feature = global_amp_feature_bin;
end

