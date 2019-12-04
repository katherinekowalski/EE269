clear all;
close all;
%%

numFrames = 10;
m = 128;
%%
X_train_wd = {};
Y_train_wd = {};
length_train = length(dir("D:\\EE269\\OutputDataTrain\\"))-2;
tr_names = {dir("D:\\EE269\\OutputDataTrain\\").name}';
for i=(round(length_train/2)+1):length_train
    p = "D:\\EE269\\OutputDataTrain\\" + string(tr_names{i+2});
    load (p);
    T=frames;
    sz = size(T);
    temp = split(p, 'class_');
    temp = split(temp(end), '.');
    label = str2double(temp(1));
    
    numSections = floor(sz(3) / numFrames);
    sample_sectioned = zeros(m,m,numFrames,numSections);
    label_sectioned = zeros(numSections, 1);
    for s=1:numSections
        label_sectioned(s) = label;
        for j=1:numFrames
            [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
            sample_sectioned(1:64,1:64,j,s) = cA;
            sample_sectioned(1:64,65:128,j,s) = cH;
            sample_sectioned(65:128,1:64,j,s) = cV;
            sample_sectioned(65:128,65:128,j,s) = cD;
        end
    end
    X_train_wd{i} = sample_sectioned;
    Y_train_wd{i} = label_sectioned;
end
X_train_wd = cat(4, X_train_wd{:});
Y_train_wd = cat(1, Y_train_wd{:});
save ("D:\EE269\train_1.mat", "X_tr1", "Y_tr1", "-v7.3")

%%
X_val_wd = {};
Y_val_wd = {};

val_names = {dir("D:\\EE269\\OutputDataVal\\").name}';

for i=1:length(dir("D:\\EE269\\OutputDataVal\\"))-2
    p = "D:\\EE269\\OutputDataVal\\" + string(val_names{i+2});
    load (p);
    T=frames;
    sz = size(T);
    temp = split(p, 'class_');
    temp = split(temp(end), '.');
    label = str2double(temp(1));
    numSections = floor(sz(3) / numFrames);
    sample_sectioned = zeros(m,m,numFrames,numSections);
    label_sectioned = zeros(numSections, 1);
    for s=1:numSections
        label_sectioned(s) = label;
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
X_val = cat(4, X_val_wd{:});
Y_val = cat(1, Y_val_wd{:});
save ("D:\EE269\val.mat", "X_val", "Y_val", "-v7.3")
%%


X_test_wd = {};
Y_test_wd = {};
test_names = {dir("D:\\EE269\\OutputDataTest\\").name}';

for i=1:length(dir("D:\\EE269\\OutputDataTest\\"))-2
    p = "D:\\EE269\\OutputDataTest\\" + string(test_names{i+2});
    load (p);
    T=frames;
    sz = size(T);
    temp = split(p, 'class_');
    temp = split(temp(end), '.');
    label = str2double(temp(1));
    numSections = floor(sz(3) / numFrames);
    sample_sectioned = zeros(m,m,numFrames,numSections);
    label_sectioned = zeros(numSections, 1);
    for s=1:numSections
        label_sectioned(s) = label;
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

X_test = cat(4, X_test_wd{:});
Y_test = cat(1, Y_test_wd{:});

save ("D:\EE269\test.mat", "X_test", "Y_test", "-v7.3")
%%
% save ('D:\EE269\wavelet_data_wd.mat','X_train_wd', 'Y_train_wd','X_val_wd', 'Y_val_wd', 'X_test_wd', 'Y_test_wd')
