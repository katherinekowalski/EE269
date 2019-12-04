% clear all; close all;
% load 'X_test.mat'
% X_test = X;
% load 'X_tr.mat'
% X_tr = X;
% load 'X_val.mat'
% X_val = X;
%%
load 'D:\EE269\test.mat'
%%
load 'D:\EE269\ytrain.mat'
Y_tr = cat(1, Y_train_wd{:});
%%
load 'D:\EE269\yval.mat'
Y_val = cat(1, Y_val_wd{:});%%
%%
load 'D:\EE269\ytest.mat'
Y_test = cat(1, Y_test_wd{:});%%
%%
load 'D:\EE269\test.mat'
%%
X_test = cat(4, X_test_wd{:});
%%
load 'D:\EE269\train.mat'
% X_test = cat(4, X_train_wd{:});
%%
load 'D:\EE269\val.mat'
X_val = cat(4, X_val_wd{:});
%%
X_t =cat(4, X_train_wd{1:200});
%%
Y_t = Y_tr(1:size(X_t, 4));
%% PCA
model = helperPCAModel(X_val,30,trainImds.Labels);
predlabels = helperPCAClassifier(testfeatures,model);
%% SVM
% nOfClassInstance=11;
% a1=reshape(X_t, [3927,128*128*10]);
% Model=svm.train(a1,Y_t);
%%
a2=reshape(X_val, [1910,128*128*10]);
predict=svm.predict(Model,a2);
%%
Accuracy=mean(Y_val==predict)*100;
fprintf('\nAccuracy =%d\n',Accuracy)