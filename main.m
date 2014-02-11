[target_variables, target_names, data, feature_names] = load_hs_vaalikone('T-61_3050_data_vk_training.csv');

% data = bsxfun(@minus,data,min(data));
% data = bsxfun(@rdivide,data,max(data));
% S = sdaeTrainBinary(data,[size(data,2) 200 100 5],32, 500, 5000);
% % S = findSdaeRBM(data,[size(data,2) 500 200 100 2]);
% H = sdae_get_hidden(data,S); % new features

[~,H] = pca(data);
H = H(:,1:23);

K = 10;
N = size(data,1);
idx = crossvalind('Kfold',N,K);
l1 = str2double(target_variables(:,1));
l2 = zeros(N,1);
party_names = unique(target_variables(:,2));
for nn = 1:N
    for kk = 1:length(party_names)
        if strcmp(party_names{kk},target_variables{nn,2})
            l2(nn,1) = kk;
            break
        end
    end
end

% feature selection for election outcome

c = cvpartition(l1,'k',10);
opts = statset('display','iter');

[fs1,history1] = sequentialfs(@crit,data,l1,'cv',c,'options',opts);

data = data(:,fs);
H = data;

% feature selection for election outcome
c = cvpartition(l2,'k',10);
opts = statset('display','iter');


[fs2,history2] = sequentialfs(@crit2,data,l2,'cv',c,'options',opts);

data = data(:,fs);
H = data;


    

% part one using SVM
sigma = [0.5 1 2 5 10];
neighb = 1:2:15;
count = 0;
res1 = zeros(N,length(sigma));
res2 = zeros(N,length(neighb));
target = zeros(N,2);
numClassifier = 101;
for kc = 1:K
    trainX = H(idx~=kc,:);
    trainL = l1(idx~=kc);
    trainC = l2(idx~=kc);
    testX = H(idx==kc,:);
    nTest = sum(idx==kc);
    target(count+1:count+nTest,:) = [l1(idx==kc) l2(idx==kc)];
%     for cc = 1:length(sigma)
%         pIdx = find(trainL == 1);
%         nIdx = find(trainL == 0);
%         nPos = length(pIdx);
%         y = zeros(size(testX,1),1);
%         for cfr = 1:numClassifier
%             permNeg = randperm(length(nIdx));
%             svm = svmtrain(trainX([pIdx; nIdx(permNeg(1:nPos))],:),[ones(nPos,1);zeros(nPos,1)],'kernel_function','rbf','rbf_sigma',sigma(cc));
%             y = y + svmclassify(svm,testX);
%             fprintf('classifier = %d\n',cfr)
%         end
%         y = y/numClassifier;
%         y = y>=(numClassifier+1)/2/numClassifier; 
%         res1(count+1:count+nTest,cc) = y;
%         fprintf('Fold = %d, Sigma = %f\n',kc,sigma(cc))
%     end
    for neighbNum = 1:length(neighb)
        mdl = ClassificationKNN.fit(trainX,trainC,'NumNeighbors',neighb(neighbNum));
        c = predict(mdl,testX);
        res2(count+1:count+nTest,neighbNum) = c;
        fprintf('Fold = %d, Num of neighbs = %d\n',kc,neighb(neighbNum))
    end
    count = count + nTest;
end

score1 = zeros(length(sigma),6);
for cc = 1:length(sigma)
    results = evaluate(res1(:,cc),target(:,1));
    score1(cc,:) = results{2};
end

[~,sigmaBestIdx] = max(score1(:,1)); 
sigmaBest = sigma(sigmaBestIdx);

score2 = zeros(length(neighb),6);
for cc = 1:length(neighb)
    results = evaluate(res2(:,cc),target(:,2));
    score2(cc,:) = mean(results{2});
end

[~,neighbBestIdx] = max(score2(:,1)); 
neighbBest = neighb(neighbBestIdx);


%% Test
[labels_1, labels_2] = textread('data_vk_test-labels_only.csv','%s %s','delimiter',',');
[~, ~, testX, ~] = load_hs_vaalikone('data_vk_test-data_only.csv');
testX = testX(:,3:end);

M = size(testX,1);
t1 = str2double(labels_1(2:end));
t2 = zeros(M,1);

labels_2 = labels_2(2:end);
for nn = 1:M
    u = labels_2{nn,1};
    for kk = 1:length(party_names)
        if strcmp(party_names{kk},u(2:end-1))
            t2(nn,1) = kk;
            break
        end
    end
end

[~,testX] = pca(testX);
testX = testX(:,1:23);

% numClassifier = 101;
% trainX = H;
% trainL = l1;
% trainC = l2;
% pIdx = find(trainL == 1);
% nIdx = find(trainL == 0);
% nPos = length(pIdx);
% y = zeros(M,1);
% for cfr = 1:numClassifier
%     permNeg = randperm(length(nIdx));
%     svm = svmtrain(trainX([pIdx; nIdx(permNeg(1:nPos))],:),[ones(nPos,1);zeros(nPos,1)],'kernel_function','rbf','rbf_sigma',1);
%     y = y + svmclassify(svm,testX); %testX(:,fs));
%     fprintf('classifier = %d\n',cfr)
% end
% y = y/numClassifier;
% y = y>=(numClassifier+1)/2/numClassifier; 

mdl = ClassificationKNN.fit(H,l2,'NumNeighbors',3);
c = predict(mdl,testX);% (:,fs));


results1 = evaluate(y,t1);
results1 = results1{2};

results2 = evaluate(c,t2);
results2 = results2{2};
