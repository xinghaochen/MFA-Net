function larfd = larfd_metric(Mg, Mgf)
% Caculate "Loss of Accuracy when Removing the Finger Differentiation"
% metric
% Mg: confusion matrix using 14 gesture classes
% Mgf: confusion matrix using 28 gesture classes
% Reference: Skeleton-based dynamic hand gesture recognition, CVPRW 2016
%   Xinghao Chen, 30 Dec, 2016

Mg_tmp = Mgf(:,1:2:end) + Mgf(:,2:2:end);
Mg_tmp2 = Mg_tmp(1:2:end,:) + Mg_tmp(2:2:end,:);
Mg_tmp2 = Mg_tmp2 ./ 2;
larfd = diag(Mg - Mg_tmp2);

end

