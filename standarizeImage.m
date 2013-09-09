% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------
%if size(im,3) > 1, im = rgb2gray(im) ; end
im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end