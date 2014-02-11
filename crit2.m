function err = crit2(Xt,yt,X,t)


mdl = ClassificationKNN.fit(Xt,yt,'NumNeighbors',1);
c = predict(mdl,X);


score = evaluate(c,t);
err = score{2};
err = 1-mean(err(:,1));