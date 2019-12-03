%% number of samples and Class initialization 
nOfSamples= ;
nOfClassInstance=11;
Y= [Y_train;Y_val];
X=[X_train;X_val];
%% SVM Classification
Model=svm.train(X,Y);
predict=svm.predict(Model,X_test);
% [Model,predict] = svm.classify(Sample,class,Sample);
disp('class predict')
disp([class predict])
%% Find Accuracy
Accuracy=mean(Y_test==predict)*100;
fprintf('\nAccuracy =%d\n',Accuracy)