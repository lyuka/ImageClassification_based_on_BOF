function [X] = sampling_for_kmeans(training, num_smp)
% ========================================================
% sample local features for unsupervised codebook training
% ========================================================

clabel = unique(training.label);
nclass = length(clabel);
num_img = length(training.label); % num of images
num_per_img = round(num_smp/num_img);   % select 'num_per_img' phow features from each image
num_smp = num_per_img*num_img;

load(training.path{1});
dimFea = size(feaSet.desc, 1);

X = single(zeros(dimFea, num_smp));
cnt = 0;

for ii = 1:num_img,
    fpath = training.path{ii};
    load(fpath);
    num_fea = size(feaSet.desc, 2);
    rndidx = randperm(num_fea);
    X(:, cnt+1:cnt+num_per_img) = feaSet.desc(:, rndidx(1:num_per_img));
    cnt = cnt+num_per_img;
end;
