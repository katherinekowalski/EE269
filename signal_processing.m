clear all;
close all;
load split_data.mat
%%
numFrames = 17;
m = 128;
X_train = zeros(m, m, numFrames-1, length(train));
X_val = zeros(m, m, numFrames-1, length(val));
X_test = zeros(m, m, numFrames-1, length(test));

OUTPUT_DATA = "OutputData";
RESULT_FILE = fullfile("data", "raw_data.mat");

for i=1:length(train)
    path = fullfile(OUTPUT_DATA, train{i});
    disp(path);
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames); %128,128,17
    for j=1:numFrames-1
        X_train(:,:,j,i) = T(:,:,j);%corr(T(:,:,j), T(:,:,j + 1));
    end
end

for i=1:length(val)
    path = fullfile(OUTPUT_DATA, val{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames); %128,128,17
    for j=1:numFrames-1
        X_val(:,:,j,i) = T(:,:,j);%corr(T(:,:,j), T(:,:,j + 1));
    end
end

for i=1:length(test)
    path = fullfile(OUTPUT_DATA, test{i});
    T=readtable(path);
    T = reshape(T{:,:}, 128,128,[]);
    T = T(:,:,1:numFrames); %128,128,17
    for j=1:numFrames-1
        X_test(:,:,j,i) = T(:,:,j);%corr(T(:,:,j), T(:,:,j + 1));
    end
end
%%
save (RESULT_FILE,'X_train', 'Y_train','X_val', 'Y_val', 'X_test', 'Y_test')
