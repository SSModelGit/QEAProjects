clearvars
close all

source = '/home/sswaminathan/olin_share/Semester2/QEA/QEAProjects/FacesModule/DataSets/curated/';

subfolders = {'angry', 'happy', 'sad', 'neutral'};

qualifier = '/simplified/';

count = 0;
for i=1:size(subfolders,2)
    count = count + size(dir([source,subfolders{i},qualifier]),1) - 3;
end

num_train = 30*size(subfolders,2);
num_test = count - num_train;
train_images = zeros([600,600,num_train]);
train_data = zeros([600*600,num_train]);
test_images = zeros([600,600,num_test]);
test_data = zeros([600*600,num_test]);
train_label = string(zeros([num_train,1]));
test_label = string(zeros([num_test,1]));

prev_spot_train = 1;
prev_spot_test = 1;
for i=1:size(subfolders,2)
    name = string(0:(size(dir([source,subfolders{i},qualifier]),1) - 3)-1);
    for j=1:size(name,2)
        imgname = [source,subfolders{i},qualifier,char(name(j)),'.png'];
        img_temp = rgb2gray(imread(imgname));
        if j < 31
            train_images(:,:,prev_spot_train) = img_temp;
            train_data(:,prev_spot_train) = reshape(img_temp,numel(img_temp),1);
            train_label(prev_spot_train) = string(subfolders{i});
            prev_spot_train = prev_spot_train + 1;
        else
            test_images(:,:,prev_spot_test) = img_temp;
            test_data(:,prev_spot_test) = reshape(img_temp,numel(img_temp),1);
            test_label(prev_spot_test) = string(subfolders{i});
            prev_spot_test = prev_spot_test + 1;
        end
    end
end

save curated.mat train_images train_data test_images test_data

train_labels = char(train_label);
test_labels = char(test_label);

save labels.mat train_labels test_labels