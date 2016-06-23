function [heatMaps, prediction] = applyModelTestPhase(test_image, param, rectangle, net, verbose)

if ~exist('verbose','var')
    verbose = 1;
end

%% Select model and other parameters from param
model = param.model(param.modelID);
boxsize = model.boxsize;
np = model.np;
nstage = model.stage;

%% Apply model, with searching thourgh a range of scales

[imageToTest, pad] = reshapeImage(test_image, rectangle, boxsize);
imageToTest = preprocess(imageToTest, 0.5, param);
    
tic;
score = applyDNN(imageToTest, net, nstage);
time = toc;
if (verbose)
    fprintf('done, elapsed time: %.3f sec\n', time);
end

pool_time = size(imageToTest,1) / size(score{1},1); % stride-8
% from 46 x 46 x 18 to 368 x 368 x 18
score = cellfun(@(x) imresize(x, pool_time), score, 'UniformOutput', false);
% from 368 x 368 x 18 to crop_h x crop_w x 18
score = cellfun(@(x) transposeToMatlabConvention(x), score, 'UniformOutput', false);
score = cellfun(@(x) imresize(x, round([rectangle(4), rectangle(3)])), score, 'UniformOutput', false);
score = cellfun(@(x) resizeIntoScaledImg(x, pad), score, 'UniformOutput', false);

%% summing the heatMaps results 
heatMaps = score;

%% generate prediction from last-stage heatMaps (most refined)
score = score{1};
prediction = zeros(np,2);
for j = 1:np
    [prediction(j,2), prediction(j,1)] = findMaximum(score(:,:,j));
end
end

function [img_out, pad] = reshapeImage(img, rectangle, boxsize)
    rectangle = round(rectangle);
    new_img = img(rectangle(2):rectangle(2)+rectangle(4),rectangle(1):rectangle(1)+rectangle(3),:);
    img_out = imresize(new_img, [boxsize, boxsize]);
    %scale = [boxsize/rectangle(3), boxsize/rectangle(4)];
    pad = zeros(1,4);
    % Order: (up, left, down, right)
    pad(1) = - rectangle(2) + 1; % crop start at rect(2) -> we have to add aboce rect(2)-1 px
    pad(2) = - rectangle(1) + 1;
    pad(3) = - (size(img,1) - rectangle(2) - rectangle(4)) - 1;
    pad(4) = - (size(img,2) - rectangle(1) - rectangle(3)) - 1;
end

function img_out = preprocess(img, mean, param)
    img_out = double(img)/256;  
    img_out = double(img_out) - mean;
    % for Matlab x are the rows and y the columns 
    % rotate columns with rows (to adapt to the caffe version)
    img_out = permute(img_out, [2 1 3]);
    
    img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    boxsize = param.model(param.modelID).boxsize;
    centerMapCell = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, param.model(param.modelID).sigma);
    img_out(:,:,4) = centerMapCell{1};
end
    
function scores = applyDNN(images, net, nstage)
    input_data = {single(images)};
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    net.forward(input_data);
    string_to_search = sprintf('stage%d', nstage);
    blob_id_C = strfind(net.blob_names, string_to_search);
    blob_id = find(not(cellfun('isempty', blob_id_C)));
    blob_id = blob_id(end);
    scores = {net.blob_vec(blob_id).get_data()};
end
    
function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
end
  
function score = transposeToMatlabConvention(score)
    score = permute(score, [2 1 3]);
end

function score = resizeIntoScaledImg(score, pad)
    np = size(score,3)-1;
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
end
    
function label = produceCenterLabelMap(im_size, x, y, sigma)
    % this function generates a gaussian peak centered at position (x,y)
    % it is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);
end