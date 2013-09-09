function [beta] = sc_approx_pooling(model, feaSet)
%================================================
% 
% Usage:
% Compute the linear spatial pyramid feature using sparse coding. 
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
% Written by Jianchao Yang @ NEC Research Lab America (Cupertino)
% Mentor: Kai Yu
% July 2008
%
% Revised May. 2010
%===============================================

B = model.vocab;
pyramid = model.numSpatialX;
gamma = 0.15;
knn = 200;

dSize = size(B, 2);
desc = feaSet.desc;
img_width = feaSet.width;
img_height = feaSet.height;

nSmp = size(desc, 2);

idxBin = zeros(nSmp, 1);

sc_codes = zeros(dSize, nSmp);

% compute the local feature for each local feature
D = desc'*B;
IDX = zeros(nSmp, knn);
for ii = 1:nSmp,
	d = D(ii, :);
	[dummy, idx] = sort(d, 'descend');
	IDX(ii, :) = idx(1:knn);
end

for ii = 1:nSmp,
    %fprintf('processing %d th column \n', ii);
    y = desc(:, ii);
    idx = IDX(ii, :);
    BB = B(:, idx);
    sc_codes(idx, ii) = feature_sign(BB, y, 2*gamma);
end

sc_codes = abs(sc_codes);

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
    xBin = ceil(feaSet.drop(1,:) / wUnit);
    yBin = ceil(feaSet.drop(2,:) / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, bId) = max(sc_codes(:, sidxBin), [], 2);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
