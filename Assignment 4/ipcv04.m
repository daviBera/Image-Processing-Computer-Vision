%% ASSIGNEMENT 4: PART 1-2-3
%  Berasi Davide

clear variables
close all
clc

% Instantiate video objects
vidReader = VideoReader('video_walk_compr.mp4');
vidWriter = VideoWriter('processed_video.mp4', 'MPEG-4');
open(vidWriter)
H = vidReader.Height; W = vidReader.width;

vidReader.CurrentTime = 0;
%% PART I
% define the filters
gauss = fspecial('gaussian', 20, 5);
sharp = [0 -1  0;
        -1  5 -1;
         0 -1  0];
sobel = fspecial('sobel');

% Original video
while vidReader.CurrentTime < 10
    % read frame and convert it to gray scale
    vidFrame = readFrame(vidReader);
    vidFrameGray = rgb2gray(vidFrame);
    
    % Original video 0 - 2
    if vidReader.CurrentTime < 2
        processedFrame = vidFrame;
        caption = 'ORIGINAL VIDEO';
    
    % Gaussian blur 2 - 4
    elseif vidReader.CurrentTime < 4
        processedFrame = imfilter(vidFrame, gauss, 'conv', 'replicate');
        caption = 'BLURRING: gaussian filter';
       
    % sharpening 4 - 6
    elseif vidReader.CurrentTime < 6
        % apply laplacian filter
        processedFrame = imfilter(vidFrame, sharp);
        caption = 'SHARPENING: filter=[0 -1  0; -1  5 -1; 0 -1  0]';

    % Sobel edge detector 6 - 8
    elseif vidReader.CurrentTime < 8
        vidFrameBlurred = imfilter(vidFrameGray, fspecial("gaussian", 10, 0.5));
        filtered_hor = imfilter(vidFrameBlurred, sobel, 'conv'); % horizontal edges
        filtered_vert = imfilter(vidFrameBlurred, sobel', 'conv'); % vertical edges
        
        T = 50; % treshold
        edgeMap_hor = (filtered_hor > T);
        edgeMap_vert = (filtered_vert > T);
        
        processedFrame = (edgeMap_hor | edgeMap_vert);
        caption = 'SOBEL EDGE DETECTOR';

    % Canny edge detector 8 - 10
    elseif vidReader.CurrentTime < 10
        edgeMap = edge(vidFrameGray, 'canny', [0.05, 0.18], 1);
        processedFrame = edgeMap;
        caption = 'CANNY EDGE DETECTOR: [0.05, 0.18], sigma=1';
    end

    % add caption to the frame
    imshow(processedFrame);
    text(10, 40, caption, 'Color', 'white', 'FontSize', 11, 'BackgroundColor', 'black'); 
    F = getframe();
    writeVideo(vidWriter, imresize(F.cdata, [H W])); % add frame to myVideo
    pause(1/vidReader.FrameRate);
end

%% PART II

% Low pass filter (smoothened disk)
[x, y] = meshgrid(-(W/2):(W/2-1), -(H/2):(H/2-1));
r = sqrt(x.^2+y.^2); R = 80;
sig = @(x) 1 ./ (1+exp(-x)); % sigmoid
lowPassOTF = sig(0.05*(-r+R)); 
lowPassOTF(lowPassOTF>0.99) = 1; lowPassOTF(lowPassOTF<0.01) = 0; % for efficiency

% High pass filter (outersmooth disk, atan(log(0.1*r)))
N = 200;
[u,v] = meshgrid(-N:(N-1), -N:(N-1));
r2 = sqrt(u.^2 + v.^2);
smf = atan(log(0.1*r2)); % smooth step function
highPassOTF = padarray(smf / max(smf(:)), [H/2-N, W/2-N], 1, 'both'); % normalize and pad with ones

while vidReader.CurrentTime < 20
    vidFrameGray = rgb2gray(readFrame(vidReader));
    ft = fft2(vidFrameGray);

    % Fourier spectrum 10 -12.5
    if vidReader.CurrentTime < 12.5
        processedFrame = log(fftshift(abs(ft))+10);
        caption = 'FOURIER SPECTRUM';

    % Low pass filter (gaussian) 12.5 - 15
    elseif vidReader.CurrentTime < 15
        processedFrame = ifft2(ft .* fftshift(lowPassOTF));
        caption = 'LOW PASS FILTER: smooth disk of ones';
    
    % High pass filter 15 - 17.5
    elseif vidReader.CurrentTime < 17.5
        processedFrame = real(ifft2(ft .* fftshift(highPassOTF)));
        caption = 'HIGH PASS FILTER: smooth disk of zeros';

    % Band pass filter: combine low & high pass 17.5 - 12
    else
        processedFrame = real(ifft2(ft .* fftshift(lowPassOTF .* highPassOTF)));
        caption = 'BAND PASS FILTER: low pass + high pass';
    end
    
    % add caption to the frame
    % add caption to the frame
    imshow(processedFrame, []);
    text(10, 40, caption, 'Color', 'white', 'FontSize', 11, 'BackgroundColor', 'black'); 
    F = getframe();
    writeVideo(vidWriter, imresize(F.cdata, [H W])); % add frame to myVideo
    pause(1/vidReader.FrameRate);
end

%% PART III

% creating the template
template = imread("template.jpg");   % read template
tpl = double(template);
tpl = tpl - mean(tpl(:)); % normalize
tplenergy = sum(tpl(:).^2); % template energy
[h, w] = size(tpl);
opticalFlow = opticalFlowLKDoG('GradientFilterSigma',1,...
     'NumFrames', 9,'ImageFilterSigma',1.5,'NoiseThreshold',0.0004);

while vidReader.CurrentTime <= 40
    vidFrame = readFrame(vidReader);
    vidFrameGray = rgb2gray(vidFrame);

    % show template 20 - 22
    if vidReader.CurrentTime < 22
        r = ceil((H-h)/2); c = ceil((W-w)/2); % template position in the frame
        processedFrame = vidFrameGray;
        processedFrame(r:(r+h-1), c:(c+w-1)) = template; 
        imshow(processedFrame);
        caption = "TEMPLATE MATCHING: template";  

    % template matching
    elseif vidReader.CurrentTime < 30.5
        im = double(vidFrameGray);
        im = im - imfilter(im, fspecial('average', size(tpl)), 'replica');
        imenergy = imfilter(im.^2, ones(size(tpl)), 'replicate');
        imcorr = imfilter(im, tpl, 'corr','replicate');
        ssd = imenergy + tplenergy - 2*imcorr;
        [minssd, I] = min(ssd, [], 'all');
        
        % show ssd 22 - 24.5
        if vidReader.CurrentTime < 24.5
            imshow(log(ssd - minssd), []);
            caption = "TEMPLATE MATCHING: sum of squared differences";

        % bounding box 24.5 - 30.5
        else
            [r, c] = ind2sub([H W], I);
            processedFrame = insertShape(vidFrame, "rectangle", ...
                [c-w/2, r-h/2, w, h], "LineWidth", 3);
            imshow(processedFrame, []);
            caption = "TEMPLATE MATCHING: bounding box";
        end

    % Optical flow  30.5 - 40
    elseif vidReader.CurrentTime < 40
        flow = estimateFlow(opticalFlow, vidFrameGray);  
        imshow(vidFrame); hold on
        h = plot(flow, 'DecimationFactor', [5 5], 'ScaleFactor', 15);
        hh = get(h,'children');  hold off
        set(hh(1),'color','r'); % color the arrows
        caption = 'OPTICAL FLOW: Lucas-Kanade algorithm';
    end

    % add frame to processed video
    text(10, 40, caption, 'Color', 'white', 'FontSize', 11, 'BackgroundColor', 'black'); 
    F = getframe();
    writeVideo(vidWriter, imresize(F.cdata, [H W])); % add frame to myVideo
    pause(1/vidReader.FrameRate);
end

%% Part IV

% Idea: given a video of coins, we want to count their total value.
% To detect the coin I will use Hough circle detection (with gaussian blur
% as preprocessing).
% The first coin detected is considered as the 'reference coin' and we
% assume to know its value. All the other detected coins are classified
% comparing the ratios (detected radius)/(reference coin detected radius)
% with the ratio (true coin radius)/(reference coin true radius).
% If the reference coin exits the frame, we change the reference coin.
% Given a detected coin, for each class c we keep track of the the number of
% frames in which the coin is classified as c, then the predicted label
% for the coin is the most frequent class.

% Istantiate video objects
vidReader = VideoReader('video_coins_compr.mp4');

% Radii of coins close to the border are poorly detected; I will update the
% radii of coins that are inside a rectangle smaller than the frame.
l = 60; xr = [l l (W-l) (W-l) l]; yr = [l (H-l) (H-l) l l]; % Rect

% Radii, labels and values for euro coins
coinsRadii = [8.1250    9.3750   10.6250    9.8750   11.1250   12.1250   11.6250   12.8750];
coinsNames = ["1c" "2c" "5c" "10c" "20c" "50c" " 1€" "2€"];
coinsValues = [0.01 0.02 0.05 0.10 0.20 0.50 1.00 2.00];

% Define pairwise distance operators
Pdist1 = @(x, y) abs(bsxfun(@minus, x, y'));
Pdist2 = @(v, w) sqrt(bsxfun(@minus, v(:,1), w(:,1)').^2 + bsxfun(@minus, v(:,2), w(:,2)').^2);

% Initialize tables where coins data will be stored
n = 100;
centers = -ones(n, 2);
radii = -ones(n, 1);
counter = zeros(n, length(coinsNames)); % counter(i,j)=#frames in which coin i is classified as class j
labels = -ones(n,1); % labels(i) = argmax(counter(i,:))
lastUpdate = -ones(n, 1); % frame number at which we updated the coin last time
insertionTime = 10000*ones(n,1); % frame number at which we inserted the coin

in = [];  % idx of the rows that correspond to coins present in the current frame
free = 1; % idx of first free row
ref = []; % idx of the row of reference coin

maxDist = 50;  % treshold for tracking
avgDepth = 10; % radii are estimated as the average over last avgDepth frames
gauss = fspecial("gaussian", 20, 2); % Gaussian kernel
numFrame = 0;
vidReader.CurrentTime = 2;
while vidReader.CurrentTime <= 22
    vidFrame = readFrame(vidReader);

    % DETECT COINS IN CURRENT FRAME
    if not(isempty(ref)) % compute interval for radius (the smaller the more accurate the detection)
        minRad = floor(0.95*radii(ref) * trueRatios(1));
        maxRad = ceil(1.05*radii(ref) * trueRatios(end));
    else
        minRad = 20; maxRad = 60;
    end
    [newCenters, newRadii] = imfindcircles( ...
        imfilter(rgb2gray(vidFrame), gauss), ...
        [minRad, maxRad], ...
        "ObjectPolarity","bright");
    
    % In first part of the video I just show detected circles
    if vidReader.CurrentTime < 6
        caption1 = sprintf("Circle detection with Hough transform");
        caption2 = "";

    % In the second part I show the coin counter
    else
        % UPDATE COINS TABLE
        
        % Assign reference coin (it is the first circle detected).
        if isempty(in) && not(isempty(newCenters))      
            centers(1,:) = newCenters(1,:);
            radii(1) = newRadii(1);
            lastUpdate(1) = numFrame; insertionTime(1) = numFrame;
            label = 2;
            counter(1, label) = 1; labels(1) = label; % reference coin is a 50cent coin (label=6)
            free = 2; in = 1; ref = 1;
            trueRatios = coinsRadii / coinsRadii(label);        
        end
    
        % compute (euclidean) distance matrix D
        if not(isempty(newCenters)) && not(isempty(in))
            D = Pdist2(centers(in,:), newCenters);
            % D(i,j) = distance from coin of index i to new detected coin of index j
        else 
            D = []; % Manage some corner cases
        end
    
        % step 1: update coins that are already classified and are detected in the new frame
        [rowMin, J] = min(D, [], 2); % get distance of closest circle among detected
        det = (rowMin < maxDist); % in(det) are the coins that are detected again in the new frame     
        centers(in(det),:) = newCenters(J(det),:); % we update their center
        % we update the radius only for the coins that are in the Rect or recently entered in the frame
        % radii are updated with the running average ove last 10 frames
        inRect = inpolygon(centers(in,1), centers(in,2), xr, yr); % coin in Rect
        justIn = numFrame-insertionTime(in) < 20; % coins that appeared less than 20 frames ago
        s = det & (inRect | justIn);
        w = min(avgDepth, numFrame - insertionTime(in(s))); % weight for averaging
        radii(in(s)) = (w.*radii(in(s))+newRadii(J(s))) ./ (w+1);
        lastUpdate(in(det)) = numFrame;

        if not(ismember(ref, in(inRect))) % if reference coin got out from the Rect, we change it
            ref = in(ceil(length(in)/2));
            trueRatios = coinsRadii / coinsRadii(labels(ref));
        end
    
        % step 2: update 'in' leaving out coins that got out ot the frame.
        % a coin is considered out of frame if it has not been updated for more than 5 frames
        gotOut = numFrame-lastUpdate(in) > 5;
        in = in(not(gotOut));
        
        % step 3: add new coins that entered in the frame
        colMin = min(D, [], 1); % for detected circles get distance from closest coin
        new = (colMin > maxDist);   
        if any(new) % store the new coins
            newIn = free:(free+sum(new)-1); % idx where we insert the new coins
            centers(newIn,:) = newCenters(new,:); 
            radii(newIn) = newRadii(new);
            lastUpdate(newIn) = numFrame; insertionTime(newIn) = numFrame;
            free = free + sum(new);
            in = cat(2, in, newIn);
        end
    
        % Step 4: classify all coins
        measuredRatios = radii(in) / radii(ref);
        ratiosDist = Pdist1(trueRatios, measuredRatios'); % size=(length(in), 8)
        [~, J] = min(ratiosDist, [], 2); % min along rows
        inRect = inpolygon(centers(in,1), centers(in,2), xr, yr); % coin in Rect
        idx = sub2ind(size(counter), in(inRect), J(inRect)');
        counter(idx) = counter(idx) + 1; % only coins in Rect
        idx2 = sub2ind(size(counter), in(not(inRect)), J(not(inRect))');
        counter(idx2) = counter(idx2) + 0.001; % for labeling coins that just entered the frame
        [~, labels] = max(counter, [], 2);
        
        % update total value
        tot = sum(coinsValues(labels(1:(free-1)))); % total sum of coins we detected so far
        
        % captions
        captions = [
            "Track coins using euclidean distance",...
            "Classify comparing size with reference coin (green one)",...
            "Need to be precise...5c and 10c radii differ by 0.25mm",...
            "Average over last 5 frames for radius estimation",...
            "Final total is correct!"
            ];
        caption1 = captions(find(vidReader.CurrentTime<[9.5 13 16.5 19.5 22.5],1));
        caption2 = sprintf("TOT: %g €", tot);

        numFrame = numFrame + 1;
    end

    % SHOW THE FRAME
    imshow(vidFrame); hold on
    % plot detected circles;
    viscircles(newCenters, newRadii,'Color', 'red');
    viscircles(centers(in,:), radii(in),'Color', 'blue');
    text(centers(in,1)-20, centers(in,2), coinsNames(labels(in)), 'Color', 'blue', 'FontSize', 20);
    % plot reference coin in green
    viscircles(centers(ref,:), radii(ref),'Color', 'green');
    text(centers(ref,1)-20, centers(ref,2), coinsNames(labels(ref)), 'Color', 'green', 'FontSize', 20);
    % add caption
    text(10, 40, sprintf("FREESTYLE: COIN COUNTER \n%s", caption1), 'Color', 'white', 'FontSize', 11);
    text(10, 100, caption2, 'Color', 'white', 'FontSize', 14);    
    hold off

    F = getframe();
    writeVideo(vidWriter, imresize(F.cdata, [H W])); % add frame to processed video
    pause(1/vidReader.FrameRate);
end
close(vidWriter);