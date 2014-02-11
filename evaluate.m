% MLPB 2013 term project classification evaluation
%
% Updated: 5.11.13 - A mistake in the calculation of the Phi score corrected.
% Updated: 1.11.13 - Some classes may be missing in the prediction. Create a square confusion matrix regardless.
%
% Data: election candidates
%
% Classification task:
% a. predict election result per candidate, 1=elected, 0=not elected
% b. predict (determine) candidate's party, several classes, denoted by
% 1,2,3 ..
%
%
% This file contains the evaluation function for the prediction task
%
% NOTE: 0/0 = 0 in the metrics.
% For multiclass, we return both the average of per-class binary-scores and 
% scores per average confusion matrix.
%

%' Evaluation metrics for predicted and test labels. Generic.
%'
%' parameters:
%' prediction Table of predictions, named data.frame
%' truth Table of true labels, same dimensions as prediction
%' 
%
% Input:
% -Two matrices prediction and (ground-)truth with same dimensions
% -Rows: samples
% -Columns: label types (e.g. elected & party => two columns).
% -Each label vector is a column in the matrix with integer values ranging from 0 to C-1, where C is the number of separate classes (e.g. 2 for elected and >2 for party)
% 
% Functionality:
% -Compute the confusion matrix between the prediction and the ground-truth
% -Compute F-score, accuracy and other metrics based on the confusion matrix
% 
% Output:
% -A cell array of performance metrics.
% -Rows: label types (matching the columns of the label matrices above)
% -For label type 'v':
%   -Entry {v,2} contains a matrix of the computed metrics
%      -For label vector with C levels it will be a (2+C)-by-6 matrix
%      -The first two rows are the average metrics computed in two different ways over the C levels
%      -The remaining C columns are the metrics computed for each of the levels separately
%      -Columns are the 6 different metrics (F-score, Phi-score, accuracy, false positive rate, sensitivity and positive predictive value)
%   -Entry {v,1} contains the rownames of the metrics matrix {v,2}
%   -Entry {v,3} contains the colnames of the metrics matrix {v,2} (name of the metric)
%
function [results]=evaluate(prediction,truth)
  results = cell(size(truth,2),3);
  % loop over variables (columns)
  for v=1:size(truth,2) 
    vals_pred = prediction(:,v);
    vals_truth = truth(:,v);
    
    levs = union(vals_pred, vals_truth);
    nlevels = length(levs);

    % Confusion matrix
    A = zeros(nlevels);
    for i=1:nlevels % Go through all levels of the prediction vector.
        for j=1:nlevels % Go through all levels of the ground-truth vector.
            A(i,j) = sum(vals_pred==levs(i) & vals_truth==levs(j)); % Compute the value(j,i) to the confusion matrix.
        end
    end
    % Used format for binary classification:
    %  TN, FN
    %  FP, TP
    %
    % If only binary classification
    if(nlevels == 2) 
      result = evaluate_binary(A);
      rownames = cell(length(levs),1);
      for ind=1:length(levs)
        rownames{ind} = num2str(levs(ind));
      end
    else
      % multi-label. We average over binary classifications
      result = zeros(nlevels,6); Bave = zeros(2,2);
      for l=1:nlevels
        % level l confusion matrix
        tp = A(l,l);
        tn = sum(sum(A([1:l-1,l+1:end],[1:l-1,l+1:end])));
        fp = sum(sum(A(l,[1:l-1,l+1:end])));
        fn = sum(sum(A([1:l-1,l+1:end],l)));
        B = [tn,fn;fp,tp];
        Bave = Bave + B;
        result(l,:) = evaluate_binary(B);
      end
      % average over the classes. Using average confusion matrix
      ave_result = evaluate_binary(Bave / nlevels);
      % and value averages
      result = [mean(result); ave_result; result];
      rownames = cell(2+length(levs),1);
      rownames{1}='score_mean';
      rownames{2}='confusion_mean';
      for ind=3:2+length(levs)
        rownames{ind} = num2str(levs(ind-2));
      end
    end
    colnames=['Fscore','phi_score','accuracy','false_pos_rate','sensitivity','pos_pred_value'];
    results{v,1}=rownames;
    results{v,2}=result; 
    results{v,3}=colnames;
  end
  % done.
%%%%%%%%%%
function [result]=evaluate_binary(A)
  % see http://en.wikipedia.org/wiki/Sensitivity_and_specificity%Worked_example
  % same as precision
  if (sum(A(2,:)))
    pos_pred_value = A(2,2)/sum(A(2,:));
  else
    pos_pred_value = 0;
  end
  sensitivity = A(2,2)/sum(A(:,2)); % same as recall
  accuracy = (A(1,1)+A(2,2))/sum(sum(A));
  false_pos_rate = A(2,1)/(A(2,1)+A(1,1));
  Fscore = 2 * (pos_pred_value*sensitivity)/(pos_pred_value + sensitivity);
  if(isnan(Fscore)),  Fscore = 0; end % we decided this.
  % phi score aka Matthews correlation coefficient
  denom = sqrt(prod(sum(A,1))*prod(sum(A,2)));
  if (denom)
    phi_score = (A(2,2)*A(1,1) - A(1,2)*A(2,1))/denom;
  else
    phi_score = 0;
  end
  % keep all
  result=[Fscore,phi_score,accuracy,false_pos_rate,sensitivity,pos_pred_value];
%%%%%%%%%%%%%%%%%%%




