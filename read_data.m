% Import the .aedat files in InputData, process to frames, and save the
% frames as csv files in OutputData.
% 
% Need to have the AedatTools package in the MATLAB path to get the 
% FramesFromEvents function.
INPUT_DIR = "D:\EE269\DvsGesture\DvsGesture";
OUTPUT_DIR_TRAIN = "D:\EE269\data\train";
OUTPUT_DIR_VAL = "D:\EE269\data\val";
OUTPUT_DIR_TEST = "D:\EE269\data\test";

for f = dir(INPUT_DIR)'
    % disp(f.name);
    spl = strsplit(string(f.name), ".");

    stem_ = string(spl(1));
    if spl(2) ~= "aedat" || strlength(stem_) < 6
        continue;
    end
    disp(stem_);
    userid = sscanf(stem_{1}(5:6), '%d');
    output_dir = OUTPUT_DIR_TRAIN;
    if userid <= 22
        continue;
    end
    if userid > 22 && userid <= 25
        output_dir = OUTPUT_DIR_VAL;
    elseif userid > 25
        output_dir = OUTPUT_DIR_TEST;
    end
    out_fn = fullfile(output_dir, spl(1) + "_class_11.csv");
    if exist(out_fn, 'file')
        disp("skipping " + spl(1) + " because it has already been processed.");
        continue;
    end
    if spl(2) == "aedat"
        read_data_file(fullfile(f.folder, f.name), fullfile(f.folder, spl(1) + "_labels.csv"), spl(1), output_dir);
    end
end



function read_data_file(input_aedat, input_csv, id, outdir)
    data = ImportAedat(struct("importParams", struct("filePath", input_aedat)));
    metadata = readmatrix(input_csv);

    FRAME_MICROSECONDS = 10000;

    for r = metadata'
        ind = find(r(2) <= data.data.polarity.timeStamp);
        ind = find(data.data.polarity.timeStamp(ind) <= r(3));
        newaedat = struct;
        newaedat.data.polarity.polarity = data.data.polarity.polarity(ind);
        newaedat.data.polarity.x = data.data.polarity.x(ind);
        newaedat.data.polarity.y = data.data.polarity.y(ind);
        newaedat.data.polarity.timeStamp = data.data.polarity.timeStamp(ind);
        newaedat.info.deviceAddressSpace = data.info.deviceAddressSpace;
        numFrames = ceil((r(3) - r(2)) / FRAME_MICROSECONDS);
        frames = FramesFromEvents(newaedat, numFrames, "time");
        disp(size(frames));
        fn = fullfile(outdir, id + "_class_" + r(1) + ".mat");
        save(fn, "frames");
    end
end

