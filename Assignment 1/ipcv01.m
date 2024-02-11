%% Question 1
clear variables
close all

% Read and show image
im = imread('car_dis.png');
im = im2double(im);
figure(1); imshow(im, []); title('Original image')

% Compute Fourier transform of the image and show it
F = fft2(im);
IM = log(abs(fftshift(F))+1);
figure(2); imshow(IM, []); title('Fourier domain representation');
xlabel('u'); ylabel('v');

% Save figure as png
print('Imlogmag', '-dpng', '-r150')

%% Question 3

% Filter the image and show it
h = fspecial('average', [1 9]);
imfil = imfilter(im, h, 'conv');
figure(3); imshow(imfil, []); title('Filtered image')
% Save the filtered image as jpg
imwrite(mat2gray(imfil), 'imfil.jpg')

% Compute the FT of the filtered image and show it
Ffil = fft2(imfil);
IMFIL = log(abs(fftshift(Ffil))+1);
figure(4); imshow(IMFIL, []);
title('Fourier domain representation of filtered image');
xlabel('u'); ylabel('v');
% Save the figure as png
print('Imfillogmag', '-dpng', '-r150')

% 3d. Remove border artifacts
imfil = imfilter(im, h, "symmetric");
figure(5); imshow(imfil, []); title('Filtered image with adjusted border')

%% Question 4

%4a. Compute OTF of the filter and show it
OTF = psf2otf(h, size(im));
magn = fftshift(abs(OTF));
% To show the OTF I linearly map the magnitudes to the interval [0, 1]
m = min(magn, [], 'all'); M = max(magn, [], 'all');
figure(3); imshow((magn-m)./(M-m)); title('OTF of the filter');
xlabel('u'); ylabel('v');
% Save the figure as png
print('OTFmag', '-dpng', '-r150')

%4b
% Imaginary part of OTF
imag_OTF = max(abs(imag(OTF(:))));
% Attenuation factor
attenuation_fac = magn(129, 157);


%% Question 5
% 5b. Create OTF matrix to cancel out noise frequencies
row = 129; col = 157; s = 15;
rstart = row - (s-1)/2; rend = row + (s-1)/2;
cstart = col - (s-1)/2; cend = col + (s-1)/2;
H = ones(size(im));
% Create square of zeros centered in (row, col)
H(rstart:rend,cstart:cend) = 0;

% To make H symmetric wrt the origin we have to insert another square of
% zeros. Thanks to its simmetry, reflecting the square wrt to the
% origin (129,129) is the same as translating it by the vector
% (-2*(row-129), -2*(col-129)),
H((rstart:rend)-2*(row-129),(cstart:cend)-2*(col-129)) = 0;

% 5c. Remove the noise frequecy in frequency domain
G = F .* fftshift(H);
imresult = ifft2(G);
imag_imresult = max(abs(imag(imresult(:))));
imresult = real(imresult);

% 5d.
% Show magnitude of the new OTF and save the figure
figure(3); imshow(H); title('Magnitude of the new OTF');
xlabel('u'); ylabel('v');
print('newOTFmag', '-dpng', '-r150')

% Show spectrum of the new filtered image and save the figure
figure(4); imshow(log(abs(fftshift(G))+1), []);
title('Filtered Fourier representation of the image');
xlabel('u'); ylabel('v');
print('imfilNewlogmag', '-dpng', '-r150')

% Show new filtered image and save it
figure(5); imshow(imresult, []); title('New filtered image')
imwrite(mat2gray(imresult), 'imfilNew.jpg')