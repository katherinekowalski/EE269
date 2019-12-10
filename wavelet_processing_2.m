clear all;
close all;
%%

in_files = {fullfile("OutputDataTrain", "user0*.mat"), fullfile("OutputDataTrain", "user1*.mat"), fullfile("OutputDataTrain", "user2*.mat") fullfile("OutputDataVal", "*.mat"), fullfile("OutputDataTest", "*.mat")};
out_dir = "in_data";
out_files = {"train_0.mat", "train_1.mat", "train_2.mat", "val.mat", "test.mat"};

numFrames = 10;
m = 128;
raw=true;
%%
for k=1:length(in_files)
    if exist(fullfile(out_dir, out_files{k}),'file')
        disp ("skipping " + fullfile(out_dir, out_files{k}) + " because the file is already present.")
        continue;
    end
    X_train_wd = {};
    Y_train_wd = {};
    files = dir(in_files{k});
    tr_names = {files.name}';
    tr_folders = {files.folder}';
    for i=1:length(files)
        p = fullfile(string(tr_folders{i}), string(tr_names{i}));
        disp (p);
        load (p);
        T=frames; % in p
        sz = size(T);
        temp = split(p, 'class_');
        temp = split(temp(end), '.');
        label = str2double(temp(1));
        
        numSections = floor(sz(3) / numFrames);
        sample_sectioned = zeros(m,m,numFrames,numSection, "single");
        label_sectioned = zeros(numSections, 1);
        for s=1:numSections
            label_sectioned(s) = label;
            for j=1:numFrames
                [cA, cH, cV, cD]=dwt2(T(:,:,j),'haar');
                sample_sectioned(1:m/2,1:m/2,j,s) = cA;
                sample_sectioned(1:m/2,m/2+1:m,j,s) = cH;
                sample_sectioned(m/2+1:m,1:m/2,j,s) = cV;
                sample_sectioned(m/2+1:m,m/2+1:m,j,s) = cD;
                % disp(sample_sectioned)
            end
        end
        X_train_wd{end+1} = sample_sectioned;
        Y_train_wd{end+1} = label_sectioned;
    end
    save (fullfile(out_dir, out_files{k}), "X_train_wd", "Y_train_wd", "-v7.3")
end
