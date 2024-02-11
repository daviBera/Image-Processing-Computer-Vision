%% Davide Berasi

clear variables
close all

%% question 1
sigma = 5; 
r=4; % as in ut_gauss
L = 2*ceil(sigma*r)+1; % set the width of the Gaussian
xymax = (L-1)/2; % the maximum of the x and the y coordinate
xrange = -xymax:xymax; % the range of x values
yrange = xrange; % the range of y values
figure(1)
h = fspecial('gaussian', L, sigma); % create the PSF matrix
C = cat(3, ones(size(h)),ones(size(h)),zeros(size(h)));
% create a RGB matrix to define the colour of the surface plot
hd = surf(xrange,yrange,h,C,'FaceColor','interp','Facelight','phong');
% create the surface plot of the gaussian
camlight right % add a light at the right side of the scene
xlim([-xymax xymax]); % set appropriate axis limits
ylim([-xymax xymax]);
xlabel('x'); % add axis labels
ylabel('y');
zlabel('h(x,y)');
print -r150 -dpng ex3_1.png % print the result to file

% 1.b
im = imread('ang2.png'); % read the image from file
sigma = [1 5 10 20]; % fill array with sigma values
for i=1:4 % do 4 times:
    h = fspecial('gaussian', L, sigma(i)); % create the PSF
    imfiltered = imfilter(im, h, 'symmetric') ; % apply the filter
    subplot(2,2,i); % define a subplot
    imshow(imfiltered,[]); % show the image
    title(['\sigma = ' num2str(sigma(i))]);% include a plot title
end
print -r150 -dpng ex3_1b.png % print the result to file

scales = sigma.^2;

%% Question 2

% 2.b Compute hy
sigma = 3;
L = 2*ceil(sigma*r)+1; % set the width of the Gaussian
xymax = (L-1)/2; % the maximum of the x and the y coordinate
xrange = -xymax:xymax; % the range of x values
yrange = xrange; % the range of y values

figure(2)
N = (L-1)/2; % get the size of half of the full range
[x,y] = meshgrid(-N:N,-N:N); % create the coordinates of a 2D orthogonal grid
hy = -y.*exp(-(x.^2 + y.^2)/(2*sigma^2)) / (2*pi*sigma^4); % d/dy of psf
C = cat(3, ones(size(hy)),ones(size(hy)),zeros(size(hy)));
hyd = surf(xrange,yrange,hy,C,'FaceColor','interp','Facelight','phong');
camlight right
xlim([-xymax xymax]); ylim([-xymax xymax]);
xlabel('x'); ylabel('y'); zlabel('h(x,y)');
print -r150 -dpng ex3_2b.png % print the result to file

% 2.c Compute OTF of hy
HY = psf2otf(hy, size(im));

% 2.d Visualize OTF as surface plot
figure(3)
L = size(HY,1);
uvmax = ceil((L-1)/2); % the maximum of the u and the v coordinate
urange = -uvmax:(uvmax-1); % the range of u values
vrange = urange;
C = cat(3, ones(size(HY)), ones(size(HY)), zeros(size(HY)));
HYd =surf(urange, vrange, imag(fftshift(HY)), C,'FaceColor','interp', ...
    'Facelight','phong', 'edgecolor', 'None');
camlight right
lim = 50;
xlim([-lim lim]); ylim([-lim lim]);
xlabel('u'); ylabel('v'); zlabel('imag(HY(u,v))');
print -r150 -dpng ex3_2d.png % print the result to file

% 2.f Analyze HY
figure()
aux = fftshift(imag(HY));
plot(urange, aux(129,:)) % RK: in HY u->columns, v->rows

% 2.g Convolute im with hy in frequency domain
IMFIL = fft2(im) .* HY; % apply filter in frequency domain
imFil = ifft2(IMFIL);  % go back to image domain
figure
imshow(imFil, []);
imwrite(mat2gray(imFil), 'ex3_2g.jpg');

%% Question 3. Computing derivatives with ut_gauss

sigma = 4;
fx = ut_gauss(im, sigma, 1, 0);
fy = ut_gauss(im, sigma, 0, 1);
fxx = ut_gauss(im, sigma, 2, 0);
fyy = ut_gauss(im, sigma, 0, 2);
fxy = ut_gauss(im, sigma, 1, 1);
diff = cat(3, fx, fy, fxx, fyy, fxy);
names = ["fx"; "fy"; "fxx"; "fyy"; "fxy"];
for i=1:5
    figure
    imshow(diff(:, :, i), []); title(names(i));
    imwrite(mat2gray(diff(:, :, i)), strcat("ex3_3_", names(i), ".jpg"));
end
clear diff


%% Question 4. Gradient and laplacian

grad_mag = sqrt(fx.^2 + fy.^2);
lap = fxx + fyy;
figure
imshow(grad_mag, []); title("gradient magnitude");
imwrite(mat2gray(grad_mag), "ex3_4_gradMag.jpg");
figure
imshow(lap, []); title("laplacian");
imwrite(mat2gray(lap), "ex3_4_lap.jpg");

%% Question 5. Marr-Hildreth’s zero crossings

mask = lap>0; % binary mask for positive laplacian
SE = strel('diamond', 1); % structuring element for erosion
maskEroded = imerode(mask, SE);
% subtract the eroded mask from the mask to get the boundary
zeroCross = mask & not(maskEroded); 

figure
imshow(mask); title("positive laplacian mask");
imwrite(mask, "ex3_5_mask.jpg");
figure
imshow(zeroCross); title("zero crossing");  
imwrite(zeroCross, "ex3_5_zeroCross.jpg");

%% Question 6

imshow(grad_mag .* zeroCross);
imcontrast(gca)
T=3.5;
edgeMap = zeroCross & (grad_mag>T);
figure
imshow(edgeMap); title(sprintf("masked gradiend    T = %0.3g", T));
imwrite(edgeMap, "ex3_6_edgeMap.jpg");

%% Hysteresis thresholding
T1 = 5.7; T2 = 0.9;
marker = zeroCross & (grad_mag>T1);
mask = zeroCross & (grad_mag>T2);

edgeMap_hyst = imreconstruct(marker, mask);
imshow(edgeMap_hyst ); title(sprintf("hysteresys    T1 = %0.3g    T2 = %0.3g", T1, T2));
imwrite(edgeMap_hyst , "ex3_6_edgeMap_hyst.jpg");

%% Question 7. ut_edge

% Canny
sigma = 3;
T1 = 0.06; T2 = 0.005;
edgeMap_c = ut_edge(im, 'c', 's', 3, 'h', [T1, T2]);
figure
imshow(edgeMap_c); title(sprintf("Canny    T1 = %0.3g    T2  = %0.3d", T1, T2));
imwrite(edgeMap_c , "ex3_7_edgeMap_c.jpg");

% Marry Hilder
T1 = 0.1; T2 = 0.004;
edgeMap_m = ut_edge(im, 'm', 's', 3, 'h', [T1, T2]);
figure
imshow(edgeMap_m); title(sprintf("Marry Hilder    T1 = %0.3g    T2  = %0.3d", T1, T2));
imwrite(edgeMap_m , "ex3_7_edgeMap_m.jpg");

%% plot edges on top of original image
figure
imshow(imoverlay(im, edgeMap_c)); title("Canny");

figure
imshow(imoverlay(im, edgeMap_m)); title("Marry Hilder");



