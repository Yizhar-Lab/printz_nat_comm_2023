% To plot the GLM data, load the .mat data file in this folder and then run the code below
tic
isConn = featureTable_preprocessed.isConn;
featureTable_preprocessed = featureTable_preprocessed(:,2:21);
selectedFeatures_nums = [1 2 3 5 8 15 16 19]; % Based on the feature selection
nCV = sum(isConn);  nShuffles = 10000;
kfoldPart = cvpartition(isConn,'KFold',nCV);
selectedFeatures_table = featureTable_preprocessed(:,selectedFeatures_nums);
selectedFeatures_names = selectedFeatures_table.Properties.VariableNames;
[crossEntropy_trueSelect,dev_trueSelect,yhat_trueSelect,beta_trueSelect] = GLM_logit_constantCV(selectedFeatures_table{:,:},logical(isConn),kfoldPart);   % True connections vs. logit prediction on true connections - only selected features
[crossEntropy_trueAll,dev_trueAll,yhat_trueAll,beta_trueAll] = GLM_logit_constantCV(featureTable_preprocessed{:,:},logical(isConn),kfoldPart);   % True connections vs. logit prediction on true connections - all features
[crossEntropy_trueSelect_noCV,~,~,~] = GLM_logit_constantCV(selectedFeatures_table{:,:},logical(isConn),[]);   % True connections vs. logit prediction on true connections - only selected features - without cross-validation
[crossEntropy_trueAll_noCV,~,~,~] = GLM_logit_constantCV(featureTable_preprocessed{:,:},logical(isConn),[]);   % True connections vs. logit prediction on true connections - all features - without cross-validation
meanPredict = ones(size(isConn))*mean(isConn);
crossEntropy_mean = mean(-isConn.*log(meanPredict)-(1-isConn).*log(1-meanPredict));  % Cross-entropy of true connections vs. mean prediction (mean connection probability for each pair)
crossEntropy_shflSelect = zeros(nShuffles,1);  % True connections vs. logit prediction on shuffled connections
dev_shflSelect = zeros(nShuffles,1);    % Calculated on the training set for each fold, while cross-entropy is based on the test set
for iShfl = 1:nShuffles
    disp(iShfl)
    shflConn = zeros(size(isConn));
    shflConn(randperm(length(isConn),sum(isConn))) = 1;
    [beta_shfl,dev_shflSelect(iShfl)] = glmfit(selectedFeatures_table{:,:},logical(shflConn),'binomial');
    yhat_shfl = glmval(beta_shfl,selectedFeatures_table{:,:},'logit');
    crossEntropy_shflSelect(iShfl) = mean(-isConn.*log(yhat_shfl)-(1-isConn).*log(1-yhat_shfl));
end
nPredictorsUsed = size(selectedFeatures_table,2);
crossEntropy_trueOmitOne = zeros(nCV,nPredictorsUsed);	% True connections vs. logit prediction on true connections (for each CV fold), where for each entry one feature is omitted
crossEntropy_trueOmitOne_noCV = zeros(1,nPredictorsUsed);
dev_trueOmitOne = zeros(nCV,nPredictorsUsed);
for iOmit = 1:nPredictorsUsed
    disp(iShfl+iOmit)
    selectedFeatures_table_omitOne = selectedFeatures_table;
    selectedFeatures_table_omitOne(:,iOmit) = [];
    [crossEntropy_trueOmitOne(:,iOmit),dev_trueOmitOne(:,iOmit),~,~] = GLM_logit_constantCV(selectedFeatures_table_omitOne{:,:},logical(isConn),kfoldPart);
    [crossEntropy_trueOmitOne_noCV(iOmit),~,~,~] = GLM_logit_constantCV(selectedFeatures_table_omitOne{:,:},logical(isConn),[]);
end
% Plotting:
allCrossEnt = [crossEntropy_trueSelect;crossEntropy_trueAll;crossEntropy_shflSelect;crossEntropy_mean;crossEntropy_trueOmitOne(:)];
step_ent = 0.0003;	bin_edges_ent = min(allCrossEnt)-step_ent:step_ent:max(allCrossEnt)+step_ent;
hist_lim = [0.105 0.123];   nCVreps = 1;
fEntropy = figure;	subplot(2,2,1); hold on
histogram(yhat_trueAll(logical(isConn)),0:0.005:1,'Normalization','probability','EdgeColor','none')
histogram(yhat_trueAll(~logical(isConn)),0:0.005:1,'Normalization','probability','EdgeColor','none')
legend('Connected pairs','Non-connected pairs')
xlabel('Regression score');  ylabel('Fraction of cell pairs')
xlim([-0.01 0.3])
title(['All features used (' num2str(size(featureTable_preprocessed,2)) ')'])
subplot(2,2,2); hold on
histogram(crossEntropy_trueSelect,bin_edges_ent,'Normalization','probability')
histogram(crossEntropy_trueAll,bin_edges_ent,'Normalization','probability')
histogram(crossEntropy_mean,bin_edges_ent,'Normalization','probability')
histogram(crossEntropy_shflSelect,bin_edges_ent,'Normalization','probability')
xlabel('Cross-entropy');    ylabel('Fraction of cases')
title('True connections vs. predictions run on true or shuffled connections, or mean prediction')
legend('True vs true (slct ftrs)','True vs true (all ftrs)','True vs mean','True vs. shfld')
ylim([0 0.1])
subplot(2,2,3); hold on
boxData = [];   boxGroup = [];
for iBox = 1:nPredictorsUsed
    boxData = [boxData;crossEntropy_trueOmitOne(:,iBox)];
    boxGroup = [boxGroup;repmat(selectedFeatures_names(iBox),size(crossEntropy_trueOmitOne(:,iBox)))];
end
boxData = [boxData;crossEntropy_trueSelect;crossEntropy_trueAll;crossEntropy_shflSelect;crossEntropy_mean];
boxGroup = [boxGroup;repmat({'OnlySlctdFtrs'},size(crossEntropy_trueSelect));
    repmat({'AllFtrs'},size(crossEntropy_trueAll));
    repmat({'shfldConnPrdctn'},size(crossEntropy_shflSelect));
    repmat({'meanConnPrdctn'},size(crossEntropy_mean))];
boxData_noCV = [crossEntropy_trueOmitOne_noCV';crossEntropy_trueSelect_noCV;crossEntropy_trueAll_noCV;
    mean(crossEntropy_shflSelect);crossEntropy_mean];
groupVars = unique(boxGroup,'stable');
p_t = zeros(size(groupVars));   % p values from t-test
for iPlot = 1:length(groupVars)
    plotData = boxData(cellfun(@(c)strcmp(c,groupVars{iPlot}),boxGroup));
    SEMdata = std(plotData)/sqrt(length(plotData)-1);
    if iPlot==nPredictorsUsed+3
        [~,p_t(iPlot)] = ttest2(crossEntropy_trueSelect,plotData);  % Unpaired - shuffles vs. cross-validated data
    else
        [~,p_t(iPlot)] = ttest(crossEntropy_trueSelect,plotData);   % Paired - same cross-validation folds; or comparing to the mean, either case is ttest
    end
    plot([iPlot iPlot],[mean(plotData)-SEMdata mean(plotData)+SEMdata],'linewidth',4,'color',[0 0 0])
    plot(iPlot,mean(plotData),'o','markersize',6,'color',[1 0 0])
    plot(iPlot,boxData_noCV(iPlot),'o','markersize',3,'color',[0 0 1])
    text(iPlot-0.4,0.114,num2str(p_t(iPlot),2),'fontsize',8)
end
xticks(1:length(groupVars));   xticklabels(groupVars);    xtickangle(45)
yticks([0.105 0.110 0.115])
plot(xlim,[mean(crossEntropy_trueSelect) mean(crossEntropy_trueSelect)],'--','color',[0 0 0])
ylabel('Cross-entropy')
subplot(2,2,4); hold on
xBarCatCERC = [{'Constant'} selectedFeatures_names];
xBarCatReorderCERC = reordercats(categorical(xBarCatCERC),xBarCatCERC);
bar(xBarCatReorderCERC,mean(beta_trueSelect,2))
errorbar(1:size(beta_trueSelect,1),mean(beta_trueSelect,2),std(beta_trueSelect,0,2)/sqrt(nCV*nCVreps-1),'linestyle','none')
ylabel('Regression coefficient')
popupmore(fEntropy)
toc