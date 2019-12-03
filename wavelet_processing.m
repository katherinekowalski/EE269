clear all;
close all;
load split_data.mat
%%

numFrames = 16;
%c_size = 64*64*4;
%X_train_2D = zeros(c_size, numFrames, length(train));
%X_val_2D = zeros(c_size, numFrames, length(val));
%X_test_2D = zeros(c_size, numFrames, length(test));
m = 128;
%X_train_3D = zeros(m/2, 2*m, numFrames, length(train));
%X_val_3D = zeros(m/2, 2*m, numFrames, length(val));
%X_test_3D = zeros(m/2, 2*m, numFrames, length(test));

X_train_wd = {};
Y_train_wd = {};
for i=1:length(train)
    path = strcat("D:\\EE269\\output_data\\", train{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    sz = size(T);
    numSections = floor(sz(3) / numFrames);
    sample_sectioned = zeros(m,m,numFrames,numSections);
    label_sectioned = zeros(numSections);
    for s=1:numSections
        label_sectioned(s) = Y_train{i};
        for j=1:numFrames
            %[c,s]=wavedec2(T(:,:,(s-1)*numFrames+j),1,'haar');
            %sample_sectioned(:,j,s) = c;
            [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
            sample_sectioned(1:64,1:64,j,s) = cA;
            sample_sectioned(1:64,65:128,j,s) = cH;
            sample_sectioned(65:128,1:64,j,s) = cV;
            sample_sectioned(65:128,65:128,j,s) = cD;
            %X_train_2D(:,j,i) = c;
            %X_train_3D(:,1:64,j,i) = cA;
            %X_train_3D(:,65:128,j,i) = cH;
            %X_train_3D(:,129:192,j,i) = cV;
            %X_train_3D(:,193:256,j,i) = cD;
        end
    end
    X_train_wd{i} = sample_sectioned;
    Y_train_wd{i} = label_sectioned;
end
X_train_wd = cat(4, X_train_wd{:});
Y_train_wd = cat(1, Y_train_wd{:});

X_val_wd = {};
Y_val_wd = {};
for i=1:length(val)
    path = strcat("D:\\EE269\\output_data\\", val{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    sz = size(T);
    numSections = floor(sz(3) / numFrames);
    sample_sectioned = zeros(m,m,numFrames,numSections);
    label_sectioned = zeros(numSections);
    for s=1:numSections
        label_sectioned(s) = Y_val{i};
        for j=1:numFrames
            [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
            sample_sectioned(1:64,1:64,j,s) = cA;
            sample_sectioned(1:64,65:128,j,s) = cH;
            sample_sectioned(65:128,1:64,j,s) = cV;
            sample_sectioned(65:128,65:128,j,s) = cD;
        end
    end
    X_val_wd{i} = sample_sectioned;
    Y_val_wd{i} = label_sectioned;
end
X_val_wd = cat(4, X_val_wd{:});
Y_val_wd = cat(1, Y_val_wd{:});

X_test_wd = {};
Y_test_wd = {};
for i=1:length(test)
    path = strcat("D:\\EE269\\output_data\\", test{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    sz = size(T);
    numSections = floor(sz(3) / numFrames);
    sample_sectioned = zeros(m,m,numFrames,numSections);
    label_sectioned = zeros(numSections);
    for s=1:numSections
        label_sectioned(s) = Y_test{i};
        for j=1:numFrames
            [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
            sample_sectioned(1:64,1:64,j,s) = cA;
            sample_sectioned(1:64,65:128,j,s) = cH;
            sample_sectioned(65:128,1:64,j,s) = cV;
            sample_sectioned(65:128,65:128,j,s) = cD;
        end
    end
    X_test_wd{i} = sample_sectioned;
    Y_test_wd{i} = label_sectioned;
end
X_test_wd = cat(4, X_test_wd{:});
Y_test_wd = cat(1, Y_test_wd{:});
%%
save ('D:\EE269\wavelet_data_wd.mat','X_train_wd', 'Y_train_wd','X_val_wd', 'Y_val_wd', 'X_test_wd', 'Y_test_wd')
