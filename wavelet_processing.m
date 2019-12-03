clear all;
close all;
load split_data.mat
%%

numFrames = 16;
c_size = 64*64*4;
X_train_2D = zeros(c_size, numFrames, length(train));
X_val_2D = zeros(c_size, numFrames, length(val));
X_test_2D = zeros(c_size, numFrames, length(test));
m = 128;
X_train_3D = zeros(m/2, 2*m, numFrames, length(train));
X_val_3D = zeros(m/2, 2*m, numFrames, length(val));
X_test_3D = zeros(m/2, 2*m, numFrames, length(test));

for i=1:length(train)
    path = strcat("D:\\EE269\\output_data\\", train{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames);
    for j=1:numFrames
        [c,s]=wavedec2(T(:,:,j),1,'haar');
        X_train_2D(:,j,i) = c;
        [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
        X_train_3D(:,1:64,j,i) = cA;
        X_train_3D(:,65:128,j,i) = cH;
        X_train_3D(:,129:192,j,i) = cV;
        X_train_3D(:,193:256,j,i) = cD;
    end
end

for i=1:length(val)
    path = strcat("D:\\EE269\\output_data\\", val{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames);
    for j=1:numFrames
        [c,s]=wavedec2(T(:,:,j),1,'haar');
        X_val_2D(:,j,i) = c;
        [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
        X_val_3D(:,1:64,j,i) = cA;
        X_val_3D(:,65:128,j,i) = cH;
        X_val_3D(:,129:192,j,i) = cV;
        X_val_3D(:,193:256,j,i) = cD;
    end
end

for i=1:length(test)
    path = strcat("D:\\EE269\\output_data\\", test{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames);
    for j=1:numFrames
        [c,s]=wavedec2(T(:,:,j),1,'haar');
        X_test_2D(:,j,i) = c;
        [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
        X_test_3D(:,1:64,j,i) = cA;
        X_test_3D(:,65:128,j,i) = cH;
        X_test_3D(:,129:192,j,i) = cV;
        X_test_3D(:,193:256,j,i) = cD;
    end
end
%%
save ('D:\EE269\wavelet_data_2D.mat','X_train_2D', 'Y_train','X_val_2D', 'Y_val', 'X_test_2D', 'Y_test')
save ('D:\EE269\wavelet_data_3D.mat','X_train_3D', 'Y_train','X_val_3D', 'Y_val', 'X_test_3D', 'Y_test')
