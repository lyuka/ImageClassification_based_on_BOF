% -------------------------------------------------------------------------
function [className, score] = classify(img_path)
% -------------------------------------------------------------------------
tStart = tic;
img = imread(img_path);
if size(img,3) > 1, img = rgb2gray(img); end
img = im2single(img);
if size(img,1) > 480, img = imresize(img, [480 NaN]); end

phowOpts = {'Sizes', 5, 'Step', 6};
patchSize = 3 * phowOpts{2};
gradSpacing = phowOpts{4};
nrml_threshold = 1;
fprintf('Dense SIFT Feature extracted from %s, patchSize = %d, gradSpacing = %d \n', img_path, patchSize, gradSpacing);
[drop, desc] = vl_phow(img, phowOpts{:});
desc = normalize_feature(desc, nrml_threshold);
feaSet = struct;
feaSet.desc = desc;
feaSet.drop = drop;
feaSet.width = size(img, 2);
feaSet.height = size(img, 1);
clear desc;
clear drop;

numWords = 200;
fprintf('Please choose corresponding ID (1~3) of coding-method \n') 
ID = input('1.psix_code; 2.sc_code; 3.LLC_code \n');
% codemethod = 'LLC_code';          % this value can be selected as 'psix_code', 'sc_code' or 'LLC_code'
switch ID
    case 1
        codemethod = 'psix_code';
    case 2
        codemethod = 'sc_code';
    case 3
        codemethod = 'LLC_code';
end
model_suffix = ['model_', codemethod, '_codebook_', num2str(numWords), '_dsift_', num2str(patchSize), '_', num2str(gradSpacing), '.mat'];
load(fullfile('data', model_suffix));
fprintf('Directly loading offline codebook. \n');
switch codemethod
    case 'psix_code'
        fprintf('Coding by VQ + SPM method. \n');
        hist = CalculateHistDescriptor(model, feaSet);
        final_code = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5);
    case 'sc_code'
        fprintf('Coding by ScSPM method. \n');
        addpath('sc_coding');
        final_code = sc_approx_pooling(model, feaSet);
    case 'LLC_code'
        fprintf('Coding by LLC method. \n');
        addpath('LLC_coding');
        final_code = LLC_pooling(model, feaSet);
end

scores = model.w' * final_code + model.b';
[score, best] = max(scores);
className = model.cname{best};
tElasped = toc(tStart);
fprintf('Image %s belongs to Class %s \n', img_path, className);
fprintf('Elasped time in the whole stage is %.2f sec. \n', tElasped);