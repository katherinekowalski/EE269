%% split data
clear all; close all;
% train-val-test, 80-10-10
nClasses = 11;
user = 0;
nusers=29;
%% 
train={};
Y_train=[];
val={};
Y_val=[];
test={};
Y_test=[];

tr_idx = 1;
val_idx = 1;
test_idx = 1;

tr_start = 1;
tr_end = floor(nusers*.8);
val_start = tr_end + 1;
val_end = val_start + floor(nusers*.2);
test_start = val_end + 1;
test_end = nusers;

tr1_start = 1;
tr1_end = floor(nClasses*.7);
val1_start = tr1_end + 1;
val1_end = val1_start + floor(nClasses*.15);
test1_start = val1_end + 1;
test1_end = nClasses;
setting= ["fluorescent", "fluorescent_led", "lab", "natural"];
for s = 1:length(setting)
    str = setting(s);
    p = randperm(nusers);
    name = strcat("D:\\EE269\\output_data\\user*_", str, "_class_%i.csv");
    for c = 1:nClasses
        a={dir(sprintf(name, c)).name};
        all= a(randperm(length(a)));
        %split into train, val, test
        for t = tr1_start:tr1_end %classes
            Y_train =[Y_train; c];
            train{tr_idx} = all{t};
            tr_idx = tr_idx + 1;
        end
        for t = val1_start:val1_end 
            val{val_idx} = all{t};;
            val_idx = val_idx + 1;
            Y_val =[Y_val; c];
        end
        for t = test1_start:test1_end 
            test{test_idx} = all{t};;
            test_idx = test_idx + 1;
            Y_test =[Y_test; c];
        end
    end
end
