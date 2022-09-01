function h = cmr(m)

%"A Color Map for Effective Black-and-white Rendering of Color-Scale Images", Carey Rappaport, IEEE Antennas and Propagation Magazine, Vol. 44, No. 3, June 2002

%CMR    Color Map Rendering for Black-and-white Images
%   CMR(M) returns an M-by-3 matrix containing a "cmr" colormap.
%   CMR, by itself, is the same length as the current figure's
%   colormap. If no figure exists, MATLAB creates one.
%
%   For example, to reset the colormap of the current figure:
%
%             colormap(cmr)
%
%   See also HSV, GRAY, PINK, COOL, BONE, COPPER, FLAG, 
%   COLORMAP, RGBPLOT.

if nargin < 1, m = size(get(gcf,'colormap'),1); end

h = [
0.00 0.00 0.00;
0.15 0.15 0.50;
0.30 0.15 0.75;
0.60 0.20 0.50;
1.00 0.25 0.15;
0.90 0.50 0.00;
0.90 0.75 0.10;
0.90 0.90 0.50;
1.00 1.00 1.00];

h=interp1(1:size(h,1),h,linspace(1,size(h,1),m));

