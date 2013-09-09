function hist = CalculateHistDescriptor(model, feaSet)
%=========================================================================
% Given phow features of an image and codebook , calculate the Hist vector
% ========================================================================

width = feaSet.width;
height = feaSet.height;
numWords = size(model.vocab, 2);

switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(feaSet.desc)), [], 1) ;    % 计算该图像中每个描述子距每个原子之间的两两距离，并找出最小的距离及相应的原子
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(feaSet.desc), ...
                                  'MaxComparisons', 15)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), feaSet.drop(1,:)) ;    % linspace在1~width区间内每隔固定距离取numSpatialX(i)+1个数，构成numSpatialX(i)个子区间；vl_binsearch则将frames(1,:)中的每个元素指派到前述各个子区间内
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), feaSet.drop(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;