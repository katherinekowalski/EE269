clear all;
close all;
load split_data.mat
%%
numFrames = 16;
m = 128;
X_train = zeros(m, m, numFrames, length(train));
X_val = zeros(m, m, numFrames, length(val));
X_test = zeros(m, m, numFrames, length(test));

for i=1:length(train)
    path = strcat("D:\\EE269\\output_data\\", train{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames);
    for j=1:numFrames
        [c,s]=wavedec2(T(:,:,j),2,'haar');
        A1 = appcoef2(c,s,'haar',1);
        X_train(:,:,j,i) = A1;
    end
end

for i=1:length(val)
    path = strcat("D:\\EE269\\output_data\\", val{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames);
    for j=1:numFrames
        [c,s]=wavedec2(T(:,:,j),2,'haar');
        A1 = appcoef2(c,s,'haar',1);
        X_val(:,:,j,i) = A1;
    end
end

for i=1:length(test)
    path = strcat("D:\\EE269\\output_data\\", test{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames);
    for j=1:numFrames
        [c,s]=wavedec2(T(:,:,j),2,'haar');
        A1 = appcoef2(c,s,'haar',1);
        X_test(:,:,j,i) = A1;
    end
end
%%
save ('D:\EE269\wavelet_data.mat','X_train', 'Y_train','X_val', 'Y_val', 'X_test', 'Y_test')