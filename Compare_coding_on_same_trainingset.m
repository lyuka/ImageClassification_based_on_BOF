imgDir = 'E:\imageLib';
dataSet = '101_ObjectCategories';
dataDir = 'data';
numWords = 200;
feature = 'dsift';
patchSize = 15;
gradSpacing = 6;
numTrain = 30;
psix_codeDir = ['psix_code_on_codebook_', num2str(numWords), '_and_', feature, '_', num2str(patchSize), '_', num2str(gradSpacing)];
sc_codeDir = ['sc_code_on_codebook_', num2str(numWords), '_and_', feature, '_', num2str(patchSize), '_', num2str(gradSpacing)];
LLC_codeDir = ['LLC_code_on_codebook_', num2str(numWords), '_and_', feature, '_', num2str(patchSize), '_', num2str(gradSpacing)];
svm_C = 10;
svm_biasMultiplier = 1;

fdatabase_path = fullfile(dataDir, ['fdatabase_', feature, '_', num2str(patchSize), '_', num2str(gradSpacing), '.mat']);
load(fdatabase_path);

codebook_path = fullfile(dataDir, ['codebook_', num2str(numWords), '_', feature, '_', num2str(patchSize), '_', num2str(gradSpacing), '_kmeans.mat']);
load(codebook_path);

subfolders = dir(fullfile(dataDir, psix_codeDir));
database.psix_code_path = {};
database.sc_code_path = {};
database.LLC_code_path = {};
for ii = 1:length(subfolders)
    subname = subfolders(ii).name;
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        frames = dir(fullfile(dataDir, psix_codeDir, subname, '*.mat'));
        c_num = length(frames);
        for jj = 1:c_num
            psix_code_path = fullfile(dataDir, psix_codeDir, subname, frames(jj).name);
            database.psix_code_path = [database.psix_code_path, psix_code_path];
            sc_code_path = fullfile(dataDir, sc_codeDir, subname, frames(jj).name);
            database.sc_code_path = [database.sc_code_path, sc_code_path];
            LLC_code_path = fullfile(dataDir, LLC_codeDir, subname, frames(jj).name);
            database.LLC_code_path = [database.LLC_code_path, LLC_code_path];
        end;    
    end;
end;
load(database.psix_code_path{1});
psix_code_dim = size(final_code, 1);
clear final_code;
load(database.sc_code_path{1});
sc_code_dim = size(final_code, 1);
clear final_code;
load(database.LLC_code_path{1});
LLC_code_dim = size(final_code, 1);
clear final_code;

selTrain = [];
selTest = [];
for ci = 1:database.nclass
    idx_label = find(database.label == ci);
    num = length(idx_label);
    idx_rand = randperm(num);
    selTrain = [selTrain, (idx_label(idx_rand(1:numTrain)))'];
    selTest = [selTest, (idx_label(idx_rand(numTrain+1:end)))'];
end

%==========================1.Test SPM======================================
psix_train_codes = single(zeros(psix_code_dim, length(selTrain)));
for jj = 1: length(selTrain)
    load(database.psix_code_path{selTrain(jj)});
    psix_train_codes(:, jj) = final_code;
end

lambda = 1 / (svm_C *  length(selTrain)) ;
model.w = [];
model.b = [];
for ci = 1:database.nclass
    perm = randperm(length(selTrain)) ;   
    fprintf('Training model for class %s \n', database.cname{ci}) ;
    y = 2 * (database.label(selTrain) == ci) - 1 ;
    [model.w(:,ci) model.b(ci)] = vl_svmtrain(psix_train_codes(:,perm), y(perm), lambda, ...
        'MaxNumIterations', 50/lambda, ...
        'BiasMultiplier', svm_biasMultiplier) ;
end
clear final_code;
clear psix_train_codes;
clear y;

result_label = [];
for jj = 1: length(selTest)
    load(database.psix_code_path{selTest(jj)});
    scores = model.w' * final_code + model.b' ;
    [score, best] = max(scores) ;
    result_label = [result_label, best]; 
    if best ~= database.label(selTest(jj))
        destination_dir = fullfile(dataDir, ['misClassify_by_psix_code']);
        if ~isdir(destination_dir)
            mkdir(destination_dir)
        end
        [a, b , c] = fileparts(database.path{selTest(jj)});
        label_id = database.label(selTest(jj));
        file_name = [database.cname{label_id}, '_' ,b, '_to_', database.cname{best}, '.jpg'];
        destination_path = fullfile(destination_dir, file_name);
        source_path = fullfile(imgDir, dataSet, database.cname{label_id}, [b, '.jpg']);
        copyfile(source_path, destination_path);
    end
end
acc_psix = zeros(database.nclass, 1);
for ci = 1: database.nclass
    idx = find(database.label(selTest) == ci);
    len = length(idx);
    acc_psix(ci) = length(find(result_label(idx) == ci)) / len;
    fprintf('Classification accuracy of %s (%d tested) by SPM is %.2f %% \n', database.cname{ci}, len, 100 * acc_psix(ci));
end
accuracy_psix = mean(acc_psix(:));
clear final_code;
clear model.w;
clear model.b;

%==========================2.Test ScSPM====================================
sc_train_codes = single(zeros(sc_code_dim, length(selTrain)));
for jj = 1: length(selTrain)
    load(database.sc_code_path{selTrain(jj)});
    sc_train_codes(:, jj) = final_code;
end

lambda = 1 / (svm_C *  length(selTrain)) ;
model.w = [];
model.b = [];
for ci = 1:database.nclass
    perm = randperm(length(selTrain)) ;   
    fprintf('Training model for class %s \n', database.cname{ci}) ;
    y = 2 * (database.label(selTrain) == ci) - 1 ;
    [model.w(:,ci) model.b(ci)] = vl_svmtrain(sc_train_codes(:,perm), y(perm), lambda, ...
        'MaxNumIterations', 50/lambda, ...
        'BiasMultiplier', svm_biasMultiplier) ;
end
clear final_code;
clear sc_train_codes;
clear y;

result_label = [];
for jj = 1: length(selTest)
    load(database.sc_code_path{selTest(jj)});
    scores = model.w' * final_code + model.b' ;
    [score, best] = max(scores) ;
    result_label = [result_label, best]; 
    if best ~= database.label(selTest(jj))
        destination_dir = fullfile(dataDir, ['misClassify_by_sc_code']);
        if ~isdir(destination_dir)
            mkdir(destination_dir)
        end
        [a, b , c] = fileparts(database.path{selTest(jj)});
        label_id = database.label(selTest(jj));
        file_name = [database.cname{label_id}, '_' ,b, '_to_', database.cname{best}, '.jpg'];
        destination_path = fullfile(destination_dir, file_name);
        source_path = fullfile(imgDir, dataSet, database.cname{label_id}, [b, '.jpg']);
        copyfile(source_path, destination_path);
    end
end
acc_sc = zeros(database.nclass, 1);
for ci = 1: database.nclass
    idx = find(database.label(selTest) == ci);
    len = length(idx);
    acc_sc(ci) = length(find(result_label(idx) == ci)) / len;
    fprintf('Classification accuracy of %s (%d tested) by ScSPM is %.2f %% \n', database.cname{ci}, len, 100 * acc_sc(ci));
end
accuracy_sc = mean(acc_sc(:));
clear final_code;
clear model.w;
clear model.b;

%==========================3.Test LLC======================================
LLC_train_codes = single(zeros(LLC_code_dim, length(selTrain)));
for jj = 1: length(selTrain)
    load(database.LLC_code_path{selTrain(jj)});
    LLC_train_codes(:, jj) = final_code;
end

lambda = 1 / (svm_C *  length(selTrain)) ;
model.w = [];
model.b = [];
for ci = 1:database.nclass
    perm = randperm(length(selTrain)) ;   
    fprintf('Training model for class %s \n', database.cname{ci}) ;
    y = 2 * (database.label(selTrain) == ci) - 1 ;
    [model.w(:,ci) model.b(ci)] = vl_svmtrain(LLC_train_codes(:,perm), y(perm), lambda, ...
        'MaxNumIterations', 50/lambda, ...
        'BiasMultiplier', svm_biasMultiplier) ;
end
clear final_code;
clear LLC_train_codes;

result_label = [];
for jj = 1: length(selTest)
    load(database.LLC_code_path{selTest(jj)});
    scores = model.w' * final_code + model.b' ;
    [score, best] = max(scores) ;
    result_label = [result_label, best]; 
    if best ~= database.label(selTest(jj))
        destination_dir = fullfile(dataDir, ['misClassify_by_LLC_code']);
        if ~isdir(destination_dir)
            mkdir(destination_dir)
        end
        [a, b , c] = fileparts(database.path{selTest(jj)});
        label_id = database.label(selTest(jj));
        file_name = [database.cname{label_id}, '_' ,b, '_to_', database.cname{best}, '.jpg'];
        destination_path = fullfile(destination_dir, file_name);
        source_path = fullfile(imgDir, dataSet, database.cname{label_id}, [b, '.jpg']);
        copyfile(source_path, destination_path);
    end
end
acc_LLC = zeros(database.nclass, 1);
for ci = 1: database.nclass
    idx = find(database.label(selTest) == ci);
    len = length(idx);
    acc_LLC(ci) = length(find(result_label(idx) == ci)) / len;
    fprintf('Classification accuracy of %s (%d tested) by LLC is %.2f %% \n', database.cname{ci}, len, 100 * acc_LLC(ci));
end
accuracy_LLC = mean(acc_LLC(:));
clear final_code;
clear model.w;
clear model.b;