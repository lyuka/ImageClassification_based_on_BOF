clear all;
clc;

conf.imgDir = 'E:\imageLib' ;
conf.phowDir = 'E:\Result' ;
conf.dataDir = 'data' ;
conf.dataSet = '101_ObjectCategories';

conf.numWords = 200;
conf.numTrain = 30;
conf.svm.C = 10;
conf.svm.biasMultiplier = 1;

conf.feature = 'dsift';            % this value can be replace with 'phow'
conf.codemethod = 'LLC_code';       % this value can be assigned as 'psix_code' , 'sc_code' or 'LLC_code'

switch conf.feature
    case 'dsift'
        conf.phowOpts = {'Sizes', 5, 'Step', 6};      % the Sizes here used in vl_phow(), and its relation to patchsize sampled is : patchsize = 4 * Sizes + 1
        patchSize = 4 * conf.phowOpts{2} + 1;
        gradSpacing = conf.phowOpts{4};
    case 'phow'
        conf.phowOpts = {'Step', 6} ;
end
switch conf.codemethod
    case 'psix_code'
        conf.codeDir = ['data\psix_code_on_codebook_', num2str(conf.numWords), '_and_', conf.feature];
    case 'sc_code'
        addpath('sc_coding');
        conf.codeDir = ['data\sc_code_on_codebook_', num2str(conf.numWords), '_and_', conf.feature];
    case 'LLC_code'
        addpath('LLC_coding');
        conf.codeDir = ['data\LLC_code_on_codebook_', num2str(conf.numWords), '_and_', conf.feature];
end

if strcmp(conf.feature, 'dsift')
    conf.codeDir = [conf.codeDir, '_', num2str(patchSize), '_', num2str(gradSpacing)];
end

rt_img_dir = fullfile(conf.imgDir, conf.dataSet);
phow_subfold = [conf.dataSet, '_', conf.feature];
if strcmp(conf.feature, 'dsift')
    phow_subfold = [phow_subfold, '_', num2str(patchSize), '_', num2str(gradSpacing)];
end
rt_phow_dir = fullfile(conf.phowDir, phow_subfold);

% Calculate Phow descriptor for the dataSet or directly Retrive it
if ~exist(fullfile(rt_phow_dir, 'airplanes'),'dir')
    tFeature_extracted_Start = tic;
    database = CalculatePhowDescriptor(rt_img_dir, rt_phow_dir, conf.phowOpts); 
    tFeature_extracted_Elapsed = toc(tFeature_extracted_Start);
    fprintf('Elapsed time in the feature extracted stage is: %.2f \n', tFeature_extracted_Elapsed);
else
    database = retr_database_dir(rt_phow_dir);
end;

% Train codebook by kmeans
vocab_suffix = ['codebook_', num2str(conf.numWords), '_', conf.feature,'_kmeans.mat'];
if strcmp(conf.feature, 'dsift')
    vocab_suffix = ['codebook_', num2str(conf.numWords), '_', conf.feature, '_', num2str(patchSize), '_', num2str(gradSpacing), '_kmeans.mat'];
end
vocab_path = fullfile(conf.dataDir, vocab_suffix);
if ~exist(vocab_path, 'file')
    num_smp = 10e4;
    descrs = sampling_for_kmeans(database, num_smp);
    fprintf('Training codebook on the above %d features by kmeans... \n',size(descrs, 2));
    vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan') ;
    fprintf('Training codebook done! \n');
    save(vocab_path, 'vocab');
else
    fprintf('Codebook already exists, loading it! \n');
    load(vocab_path);
end

clear descrs;

model.cname = database.cname; 
model.quantizer = 'kdtree';
if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end
model.vocab = vocab;
clear vocab;

model.numSpatialX = [1 2 4];
model.numSpatialY = [1 2 4];
if length(model.numSpatialX) == 1
    conf.codeDir = [conf.codeDir, '_single'];
end
% Calculate final image descriptor for the dataSet
if ~exist(fullfile(conf.codeDir, 'airplanes'),'dir')
    tFeature_Coding_Start = tic;    
    database.code_path = {};
    for ii = 1: database.imnum
        phow_path = database.path{ii};
        [a, b, c] = fileparts(phow_path);
        load(phow_path);
        switch conf.codemethod
            case 'psix_code'
                hist = CalculateHistDescriptor(model, feaSet);
                final_code = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5);
            case 'sc_code'
                final_code = sc_approx_pooling(model, feaSet);
            case 'LLC_code'
                final_code = LLC_pooling(model, feaSet);
        end
        for jj = 1: database.nclass
            if database.label(ii) == jj
                fprintf('%s vector of %s done! Now save it. \n', conf.codemethod, fullfile(database.cname{jj}, [b, '.jpg']));
                code_dir = fullfile(conf.codeDir, database.cname{jj});
                if ~isdir(code_dir)
                    mkdir(code_dir);
                end;
                code_path = fullfile(code_dir, [b, c]);
                save(code_path, 'final_code');
                database.code_path = [database.code_path, code_path];
            end;
        end;
    end;
    %size(hist);
    rep_dim = size(final_code, 1);
    clear feaSet;
    clear hist;
    clear final_code;
    tFeature_Coding_Elapsed = toc(tFeature_Coding_Start);
    fprintf('Elapsed time in the feature coding stage is: %.2f \n', tFeature_Coding_Elapsed);
else
    fprintf('The final code already exists! Directly load it when use.\n');
    subfolders = dir(conf.codeDir);
    database.code_path = {};
    for ii = 1:length(subfolders)
        subname = subfolders(ii).name;
        if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
            frames = dir(fullfile(conf.codeDir, subname, '*.mat'));
            c_num = length(frames);
            for jj = 1:c_num
                code_path = fullfile(conf.codeDir, subname, frames(jj).name);
                database.code_path = [database.code_path, code_path];
            end;    
        end;
    end;
    load(database.code_path{1});
    rep_dim = size(final_code, 1);
    clear final_code;
end

% Traing linear svm and test its performance
nRounds = 5;
acc = zeros(database.nclass, nRounds);
accuracy = zeros(nRounds, 1);
for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tSVM_Training_Start = tic;
    selTrain = [];
    selTest = [];
    for ci = 1:database.nclass
        idx_label = find(database.label == ci);
        num = length(idx_label);
        idx_rand = randperm(num);
        selTrain = [selTrain, (idx_label(idx_rand(1:conf.numTrain)))'];
        selTest = [selTest, (idx_label(idx_rand(conf.numTrain+1:end)))'];
    end
    train_codes = single(zeros(rep_dim, length(selTrain)));
    for jj = 1: length(selTrain)
        load(database.code_path{selTrain(jj)});
        train_codes(:, jj) = final_code;
    end
    lambda = 1 / (conf.svm.C *  length(selTrain)) ;
    model.w = [];
    model.b = [];
    for ci = 1:database.nclass
        perm = randperm(length(selTrain)) ;   
        fprintf('Training model for class %s \n', database.cname{ci}) ;
        y = 2 * (database.label(selTrain) == ci) - 1 ;
        [model.w(:,ci) model.b(ci)] = vl_svmtrain(train_codes(:,perm), y(perm), lambda, ...
                                        'MaxNumIterations', 50/lambda, ...
                                        'BiasMultiplier', conf.svm.biasMultiplier) ;
    end
    clear final_code;
    clear train_codes;
    tSVM_Training_Elapsed = toc(tSVM_Training_Start);
    fprintf('Elapsed time in the training svm classifier stage is: %.2f \n', tSVM_Training_Elapsed);
    
    tSVM_Testing_Start = tic;
    result_label = [];
    for jj = 1: length(selTest)
        load(database.code_path{selTest(jj)});
        scores = model.w' * final_code + model.b' ;
        [score, best] = max(scores) ;
        result_label = [result_label, best]; 
        if best ~= database.label(selTest(jj))
            destination_dir = fullfile(conf.dataDir, ['misClassify_by_', conf.codemethod, '_on_round_', num2str(ii)]);
            if ~isdir(destination_dir)
                mkdir(destination_dir)
            end
            [a, b , c] = fileparts(database.path{selTest(jj)});
            label_id = database.label(selTest(jj));
            file_name = [database.cname{label_id}, '_' ,b, '_to_', database.cname{best},'.jpg'];
            destination_path = fullfile(destination_dir, file_name);
            source_path = fullfile(conf.imgDir, conf.dataSet, database.cname{label_id},[b, '.jpg']);
            copyfile(source_path, destination_path);
        end
    end
    
    for ci = 1: database.nclass
        idx = find(database.label(selTest) == ci);
        len = length(idx);
        acc(ci, ii) = length(find(result_label(idx) == ci)) / len;
        fprintf('Classification accuracy of %s (%d tested) in Round %d is %.2f %% \n', database.cname{ci}, len, ii, 100 * acc(ci, ii));
    end
    accuracy(ii) = mean(acc(:, ii));
    tSVM_Testing_Elapsed = toc(tSVM_Testing_Start);
    fprintf('Elapsed time in the testing svm classifier stage is: %.2f \n', tSVM_Testing_Elapsed);
end

fprintf('Mean accuracy: %.2f %% \n', 100 * mean(accuracy));
fprintf('Standard deviation: %.2f %%\n', 100 * std(accuracy));

model_suffix = ['model_', conf.codemethod, '_codebook_', num2str(conf.numWords), '_', conf.feature,'.mat'];
if strcmp(conf.feature, 'dsift')
    model_suffix = ['model_', conf.codemethod, '_codebook_', num2str(conf.numWords), '_', conf.feature, '_', num2str(patchSize), '_', num2str(gradSpacing), '.mat'];
end
if length(model.numSpatialX) == 1
    model_suffix = ['model_single.mat'];
end
save(fullfile(conf.dataDir, model_suffix), 'model');
