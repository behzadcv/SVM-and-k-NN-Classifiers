function err = crit(Xt,yt,X,t)

numClassifier = 11;
pIdx = find(yt == 1);
nIdx = find(yt == 0);
nPos = length(pIdx);
y = zeros(length(t),1);
for cfr = 1:numClassifier
    permNeg = randperm(length(nIdx));
    svm = svmtrain(Xt([pIdx; nIdx(permNeg(1:nPos))],:),[ones(nPos,1);zeros(nPos,1)]);
    y = y + svmclassify(svm,X);
end
y = y/numClassifier;
y = y>=(numClassifier+1)/2/numClassifier;


score = evaluate(y,t);
err = score{2};
err = 1-err(1);