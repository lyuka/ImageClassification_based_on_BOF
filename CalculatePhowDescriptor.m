function database = CalculatePhowDescriptor(rt_img_dir, rt_data_dir, phowOpts)
%==========================================================================
% usage: calculate the sift descriptors given the image directory
%
% inputs
% rt_img_dir    -image database root path
% rt_data_dir   -phow feature database root path
% phowOpts      -for vl_phow() function
%
% outputs
% database      -directory for the calculated sift features
%
% SIFT code in VL_feat library is used.
%==========================================================================

disp('Extracting Phow features...');
subfolders = dir(rt_img_dir);

siftLens = [];
database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

nrml_threshold = 1;
for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_img_dir, subname, '*.jpg'));
        
        c_num = length(frames);           
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        siftpath = fullfile(rt_data_dir, subname);        
        if ~isdir(siftpath),
            mkdir(siftpath);
        end;
        
        for jj = 1:c_num,
            imgpath = fullfile(rt_img_dir, subname, frames(jj).name);
            
            I = imread(imgpath);
            I = standarizeImage(I);
            width = size(I,2) ;
            height = size(I,1) ;
            fprintf('Extracting phow features on %s... \n', fullfile(subname, frames(jj).name));

            % find SIFT descriptors
            [drop, desc] = vl_phow(I, phowOpts{:}) ;
            %siftArr = sp_find_sift_grid(I, gridX, gridY, patchSize, 0.8);
            desc = normalize_feature(desc, nrml_threshold);
            
            %siftLens = [siftLens, siftlen];
            
            feaSet.desc = desc;
            feaSet.drop = drop;
            feaSet.width = width;
            feaSet.height = height;
            
            [pdir, fname] = fileparts(frames(jj).name);                        
            fpath = fullfile(rt_data_dir, subname, [fname, '.mat']);
            
            save(fpath, 'feaSet');
            database.path = [database.path, fpath];
        end;    
    end;
end;

%lenStat = hist(siftLens, 100);
