function [crossEntropy,dev,yhat,beta] = GLM_logit_constantCV(features,y,CVfolds)
% Performs unregularized regression using logistic link function on binary (logical) response data y from the features.
% Uses stratified k-fold cross-validation using the pre-partitioning of the data y appearing in CVfolds,
% and calculates for each fold the cross-entropy between the data y and the prediction.
% If CVfolds is empty, performs no cross-validation.
% Also returns the overall prediction (yhat) and the coefficient for every predictor for each fold (beta).

if ~isempty(CVfolds)
    nFolds = CVfolds.NumTestSets;
    crossEntropy = zeros(nFolds,1);  dev = zeros(nFolds,1);
    yhat = NaN(size(y));	% NaNs and not zeros in case there are y entries that do not appear in any test set/fold
    beta = zeros(size(features,2)+1,nFolds);
    for i = 1:nFolds
        [beta(:,i),dev(i)] = glmfit(features(training(CVfolds,i),:),y(training(CVfolds,i)),'binomial');
        yhat_test = glmval(beta(:,i),features(test(CVfolds,i),:),'logit');	% Prediction using coefficients from fitting on the training set and predictors of the test set
        crossEntropy(i) = mean(-y(test(CVfolds,i)).*log(yhat_test)-(1-y(test(CVfolds,i))).*log(1-yhat_test));
        yhat(test(CVfolds,i)) = yhat_test;
    end
else
    [beta,dev] = glmfit(features,y,'binomial');
    yhat = glmval(beta,features,'logit');
    crossEntropy = mean(-y.*log(yhat)-(1-y).*log(1-yhat));
end
if any(isnan(yhat))
    error('yhat contains NaNs')
end

end
