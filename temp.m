% load 'D:\EE269\val.mat'
% %%
% load 'D:\EE269\test.mat'
% %%
% load 'D:\EE269\train.mat'
% %%
% load 'D:\EE269\ytrain.mat'

for i= [50,80, 600]
    X_train = X_train_wd{i};
    figure
    imagesc(X_train(:,:,2,1))
    Y_train = Y_train_wd{i};
    title(num2str(Y_train(1)));
end
%%
for i=1:10
    figure
    imagesc(X_test(:,:,i,1000));
end
Y_test(1000)
%%
% T=readtable('D:\EE269\output_data\user24_fluorescent_class_2.csv');
% T = reshape(T{:,:}, 128,128,[]);
load 'D:\EE269\data\train\user01_fluorescent_class_10.mat'

close all
% a=size(frames);
% X=zeros([a(1:2),1,a(3)]);
figure
idx=1;
for i= 6:2:24%30:-1:1
    subplot(1,6,idx);
    idx=idx+1;
    imagesc(frames(:,:,i))
    colormap gray
%     X(:,:,1,i) = imagesc(frames(:,:,i));
%     t =X(:,:,1,i);
%     a_min=min(min(t));
%     X(:,:,1,i) = 1+255*(t-a_min)/max(max(t-a_min));
end
title('Air Guitar')
% mov = immovie(X,gray);
% implay(mov)