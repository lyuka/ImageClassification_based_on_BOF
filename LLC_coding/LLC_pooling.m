function [beta] = LLC_pooling(model, feaSet)
%================================================
% 
% Usage:
% Pooling the llc codes to form the image feature 
%
% Inputss:
%   feaSet        -structure defining the feature set of an image   
%                       .desc     local feature array extracted from the
%                                   image, column-wise
%                       .drop     x and y locations of each local feature, 2nd
%                                   dimension of the matrix
%                       .width      width of the image
%                       .height     height of the image
%   model         
%                       .vocab         -dictionary, column-wise
%                       .numSpatialX   -defines structure of pyramid 
% 
% Output:
%   beta          -multiscale max pooling feature
%
% Written by Jianchao Yang @ IFP UIUC
% May, 2010
%===============================================

B = model.vocab;
pyramid = model.numSpatialX;
dSize = size(B, 2);
nSmp = size(feaSet.desc, 2);
knn = 5;

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llc coding
llc_codes = LLC_coding_appr(B', feaSet.desc', knn);
llc_codes = llc_codes';

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.drop(1, :) / wUnit);
    yBin = ceil(feaSet.drop(2, :) / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
