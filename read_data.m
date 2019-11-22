function read_data
% Import the .aedat files in InputData, process to frames, and save the
% frames as csv files in OutputData.
% 
% Need to have the AedatTools package in the path to get the 
% FramesFromEvents function.
    INPUT_DIR = "InputData";
    OUTPUT_DIR = "OutputData";

    for f = dir(INPUT_DIR)'
%         [filepath, name, ext] = fileparts(f.name);
        disp(f.name);
        spl = strsplit(string(f.name), ".");
        if spl(2) == "aedat"
            read_data_file(fullfile(f.folder, f.name), fullfile(f.folder, spl(1) + "_labels.csv"), spl(1), OUTPUT_DIR);
        end
    end
end

function read_data_file(input_aedat, input_csv, id, outdir)
    data = ImportAedat(struct("importParams", struct("filePath", input_aedat)));
    metadata = readmatrix(input_csv);

    FRAME_MICROSECONDS = 100000;

    for r = metadata'
        ind = find(r(2) <= data.data.polarity.timeStamp <= r(3));
        newaedat = struct;
        newaedat.data.polarity.polarity = data.data.polarity.polarity(ind);
        newaedat.data.polarity.x = data.data.polarity.x(ind);
        newaedat.data.polarity.y = data.data.polarity.y(ind);
        newaedat.data.polarity.timeStamp = data.data.polarity.timeStamp(ind);
        newaedat.info.deviceAddressSpace = data.info.deviceAddressSpace;
        numFrames = ceil((r(3) - r(2)) / FRAME_MICROSECONDS);
        frames = FramesFromEvents(newaedat, numFrames, "time");
        writematrix(frames, fullfile(outdir, id + "_class_" + r(1) + ".csv"));
    end
end


