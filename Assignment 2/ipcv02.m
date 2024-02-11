%% Assigment 2: Image registration and geometrical transforms %%

clear variables
close all

%% Question 1. Read the images
images = imageSet('images');

% for i=1:4
%     figure; imshow(read(images, i)); title(sprintf('Image %d', i));
% end

%% Question 3. Pinpoint corresponding points
% for n=1:4
%     for m=n+1:4
%         [p{n,m},p{m,n}] = cpselect(read(images,n), read(images,m), 'Wait', true);
%     end
% end
% 
% save psets p;       % Save the corresponding points

%% PART I: stich im2 to im1

%% Question 4. Estimating the geometric transform
load psets;           % Load the corresponding points

% p{n,m} is a Kx2 array that contains the coords of the points in image n
% that correspond to a point in image m

im1 = read(images, 1);
im2 = read(images, 2);

% Compute the projective transform that maps im2 -> im1
tform2 = estimateGeometricTransform(p{2,1}, p{1,2}, 'projective');
im2to1 = imwarp(im2, tform2);

figure; imshow(im2to1); title('im2 warped to im1');
imwrite(im2to1, 'im2to1.jpg')   % Save warped image

im1_size = size(im1);
im2to1_size = size(im2to1);

%% Question 5. Define the world coordinate system and stitch images
% All 4 images have the same size
height = im1_size(1); width = im1_size(2); 

% Limits of im1 after transform
tform1 = projective2d(eye(3));    % Identity operator
[xlims(1,:), ylims(1,:)] = outputLimits(tform1, [1 width], [1 height]);

% Limits of im2 after transform
[xlims(2,:), ylims(2,:)] = outputLimits(tform2, [1 width], [1 height]);

% World limits for stitched images
xMin = min([1; xlims(:)]);
xMax = max([width; xlims(:)]);
yMin = min([1; ylims(:)]);
yMax = max([height; ylims(:)]);

xWorldLimits = [xMin xMax];
yWorldLimits = [yMin yMax];

% Width and height of final image.
width12  = round(xMax - xMin);
height12 = round(yMax - yMin);

% World coordinate system
imref = imref2d([height12 width12], xWorldLimits, yWorldLimits);

% Stitching images:
% Initialize final image and blender object.
stitched12 = zeros([height12 width12 3], 'like', im1);
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Transform im1 into the final image.
im1to1 = imwarp(im1, tform1, 'OutputView', imref);
% Generate a binary mask.
mask1 = imwarp(true(height, width), tform1, 'OutputView', imref); 
% Overlay the warpedImage onto the final image.
stitched12 = step(blender, stitched12, im1to1, mask1);

% Same procedure for im2
im2to1 = imwarp(im2, tform2, 'OutputView', imref);
mask2 = imwarp(true(height, width), tform2, 'OutputView', imref);
stitched12 = step(blender, stitched12, im2to1, mask2);

figure; imshow(stitched12); title('im2 and im1 stitched')
imwrite(stitched12, 'stitched12.jpg')

%% PART II: sequantially stiching the 4 images

%% Question 6. Construct tform and tform1

% Compute tform
tform = projective2d.empty(4,0);
tform(1) = projective2d(eye(3));
for n=2:4
    tform(n) = estimateGeometricTransform(p{n,n-1}, p{n-1,n}, 'projective');
end

% Compute tform1
tform1(1) = tform(1);
tform1(2) = tform(2);

% Now I compute the points in the CS of im1:
% p1{m,n}(k) is p{m,n}(k) in CS of image 1
%
% p1{3,2} = (tform(2) o tform(3))( p{3,2} ) 
% p1{4,3} = (tform(2) o tform(3) o tform(4))( p{4,3} )
% RK: For a neater code I don't exploit the fact that
%     tform(n)( p{n,n-1}(k) ) = p{n-1,n}(k)
p1 = p;
for n = 3:4
    for i=n:-1:2
        [p1{n,n-1}, p1{n,n-1}(:,2)] = transformPointsForward( ...
            tform(i), p1{n,n-1}(:,1), p1{n,n-1}(:,2));
    end
    tform1(n) = estimateGeometricTransform(p{n,n-1}, p1{n, n-1}, 'projective');
end

%% Question 7. Stiching the 4 images

% Limits of the 4 images after transform
for n=1:4
    [xlims(n,:), ylims(n,:)] = outputLimits(tform1(n), [1 width], [1 height]);
end

% World limits for stitched images
xMin = min([1; xlims(:)]);
xMax = max([width; xlims(:)]);
yMin = min([1; ylims(:)]);
yMax = max([height; ylims(:)]);

xWorldLimits = [xMin xMax];
yWorldLimits = [yMin yMax];

% Width and height of final image.
widthF  = round(xMax - xMin);
heightF = round(yMax - yMin);

imref = imref2d([heightF widthF], xWorldLimits, yWorldLimits);

% Stitching images:
% (For that I followed the example 'Feature Based Panoramic Image Stitching')

% Initialize the final image.
stitchedF = zeros([heightF widthF 3], 'like', im1);
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

for n=1:4
    % Transform im into the final image.
    imnto1 = imwarp(read(images, n), tform1(n), 'OutputView', imref);
    % Generate a binary mask.
    mask = imwarp(true(height, width), tform1(n), 'OutputView', imref); 
    % Overlay the warpedImage onto the final image.
    stitchedF = step(blender, stitchedF, imnto1, mask);
end

stitchedF_size = size(stitchedF);
figure; imshow(stitchedF); title('All 4 images stitched toghether');
imwrite(stitchedF, 'stitchedF.jpg');

%% Question 8. Measuring accuracy

% Compute all point in im1 CS (I recompute all of them for a neater code)
for n=1:4
    for m=1:4
        if m ~= n
            [p1{n,m}, p1{n,m}] = transformPointsForward( ...
                tform1(n), p{n,m}(:,1), p{n,m}(:,2));
        end
    end
end

% Compute pairwise RMS matrix E (RK: it's symmetric)
e = cell(4);
E = zeros(4,4); 
for n=1:4
    for m=(n+1):4
        K = size(p1{n,m}, 1);
        e{n,m} = vecnorm(p1{m,n} - p1{n,m}, 2, 2);
        E(n,m) = norm(e{n,m}) / sqrt(K);
    end
end

% Compute overall RMS
RMS = norm(E) / sqrt(6);

%% Question 9. Rectifiing the image

% Get the landmarks
% The tool getpts gives me some problems when zooming in.
% So I used the tool Data Tips to get the ladmarks' coordinates.
% In minutes my points are:
%   (1, 22) --- (24, 22)
%      |           |
%   (1, 8)  --- (24, 8)

% In pixels they are:
x = [1447 233 319 1402];
y = [487 472 1770 1810];

landmarks = [x', y']; % 4x2

% Show selected points on image
hold on; plot(x, y,'rx', LineWidth=2); hold off;

% Compute sizes of the rectangle in nautic miles
lat = 53.2;
Dx = 21*cos(lat*pi/180); Dy = 14;
% Convert it to pixels
Dx = 100*Dx; Dy = 100*Dy;

% Compute coordinates of the rectangle
% RK: the top_left corner coincides with the top_left landmark
tr = [x(2) + Dx, y(2)];
tl = [x(2), y(2)];
bl = [x(2), y(2) + Dy];
br = [x(2) + Dx, y(2) + Dy];
corners = [tr; tl; bl; br];

% Estimate geometric transform. Rectify and crop the image
rectifier = estimateGeometricTransform(landmarks, corners, 'projective');
im_rect = imwarp(stitchedF, rectifier);
figure; imshow(im_rect); title('Rectified image')

% im_crop = imcrop(im_rect);
% imwrite(im_crop, 'im_crop.jpg');

im_crop = imread('im_crop.jpg');
figure; imshow(im_crop); title('Final image')
