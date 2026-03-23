function main()
clear; close all; clc;

%% ============================================================
% TP RECONNAISSANCE CODE POSTAL
% REPONSE STRUCTUREE QUESTION PAR QUESTION
% Compatible Octave / MATLAB
% ============================================================

cfg.trainDir = fullfile(pwd, 'data', 'train');
cfg.testDir  = fullfile(pwd, 'data', 'test');

cfg.minObjectArea = 20;
cfg.expectedDigits = 5;
cfg.resizeDigitTo = [64 64];
cfg.k = 3;
cfg.classifier = 'knn';

%% ============================================================
% QUESTION 1
% Construire Xtrain et Ytrain (apprentissage hors ligne)
% ============================================================

fprintf('\n=== QUESTION 1 : APPRENTISSAGE ===\n');

[Xtrain, Ytrain] = build_training_set(cfg);

fprintf('Nombre d''echantillons : %d\n', size(Xtrain,1));

if isempty(Xtrain)
    error('Aucun echantillon d''apprentissage n''a ete extrait.');
end

%% ============================================================
% QUESTION 2
% Calcul des moyennes par classe
% ============================================================

fprintf('\n=== QUESTION 2 : MOYENNES PAR CLASSE ===\n');

classMeans = compute_class_means(Xtrain, Ytrain);

disp('Moyennes calculees.');

%% ============================================================
% NORMALISATION
% ============================================================

mu = mean(Xtrain, 1);
sigma = std(Xtrain, 0, 1);
sigma(sigma < eps) = 1;

XtrainNorm = (Xtrain - mu) ./ sigma;
classMeansNorm = (classMeans - mu) ./ sigma;

model.XtrainNorm = XtrainNorm;
model.Ytrain = Ytrain;
model.classMeansNorm = classMeansNorm;
model.mu = mu;
model.sigma = sigma;
model.k = cfg.k;

%% ============================================================
% QUESTION 3
% Classification en ligne (un code postal)
% ============================================================

fprintf('\n=== QUESTION 3 : TEST SUR IMAGE ===\n');

fileList = dir(fullfile(cfg.testDir, '*.png'));

for i = 1:length(fileList)
    imgPath = fullfile(fileList(i).folder, fileList(i).name);
    predCode = recognize_postal_code(imgPath, model, cfg);
    fprintf('%s -> %s\n', fileList(i).name, predCode);
end

%% ============================================================
% QUESTION 4
% Comparaison PPV vs plus proche moyenne
% ============================================================

fprintf('\n=== QUESTION 4 : COMPARAISON ===\n');

cfg.classifier = 'knn';
res_knn = evaluate(cfg, model);

cfg.classifier = 'mean';
res_mean = evaluate(cfg, model);

fprintf('KNN accuracy : %.2f %%\n', 100*res_knn);
fprintf('Mean accuracy : %.2f %%\n', 100*res_mean);

%% ============================================================
% QUESTION 5
% Discussion (à mettre dans le rapport)
% ============================================================
% -> A FAIRE DANS LE PDF, PAS DANS LE CODE

%% ============================================================
% QUESTION 6
% Test avec base indépendante
% ============================================================

fprintf('\n=== QUESTION 6 : TEST BASE ===\n');

cfg.classifier = 'knn';
acc = evaluate(cfg, model);

fprintf('Accuracy test : %.2f %%\n', 100*acc);

%% ============================================================
% QUESTION 7
% Ajout de descripteurs invariants (regionprops)
% ============================================================
% -> deja inclus dans extract_features()

%% ============================================================
% QUESTION 8
% Correction de rotation
% ============================================================
% -> non implementee explicitement ici

%% ============================================================
% QUESTION 9
% Separation des chiffres collés
% ============================================================
% -> segmentation simple par composantes connexes

end

%% ============================================================
% CONSTRUCTION DATASET
%% ============================================================

function [Xtrain, Ytrain] = build_training_set(cfg)

files = dir(fullfile(cfg.trainDir, '*.png'));

Xtrain = [];
Ytrain = [];

for i = 1:length(files)

    filePath = fullfile(files(i).folder, files(i).name);

    % la classe est le premier caractere du nom de fichier
    digitClass = str2double(files(i).name(1));

    if isnan(digitClass)
        continue;
    end

    I = imread(filePath);
    digits = segment_digits(I, cfg);

    for k = 1:length(digits)
        feat = extract_features(digits{k});
        Xtrain(end+1,:) = feat; %#ok<AGROW>
        Ytrain(end+1,1) = digitClass; %#ok<AGROW>
    end
end

end

%% ============================================================
% MOYENNES
%% ============================================================

function means = compute_class_means(X, Y)

means = zeros(10, size(X,2));

for c = 0:9
    idx = (Y == c);
    if any(idx)
        means(c+1,:) = mean(X(idx,:), 1);
    end
end

end

%% ============================================================
% RECONNAISSANCE
%% ============================================================

function code = recognize_postal_code(imgPath, model, cfg)

I = imread(imgPath);
digits = segment_digits(I, cfg);

code = '';

for i = 1:length(digits)

    feat = extract_features(digits{i});
    feat = (feat - model.mu) ./ model.sigma;

    if strcmp(cfg.classifier, 'knn')
        label = knn(feat, model);
    else
        label = nearest_mean(feat, model);
    end

    code = [code num2str(label)]; %#ok<AGROW>
end

end

%% ============================================================
% KNN
%% ============================================================

function label = knn(x, model)

D = sum((model.XtrainNorm - x).^2, 2);
[~, idx] = sort(D, 'ascend');

k = min(model.k, length(idx));
label = mode(model.Ytrain(idx(1:k)));

end

%% ============================================================
% PLUS PROCHE MOYENNE
%% ============================================================

function label = nearest_mean(x, model)

D = sum((model.classMeansNorm - x).^2, 2);
[~, idx] = min(D);
label = idx - 1;

end

%% ============================================================
% SEGMENTATION
%% ============================================================

function digits = segment_digits(I, cfg)

I = rgb2gray_if_needed(I);

% Binarisation compatible Octave
level = graythresh(I);
BW = im2bw(I, level);

% On veut les chiffres en blanc
if mean(BW(:)) > 0.5
    BW = ~BW;
end

BW = bwareaopen(BW, cfg.minObjectArea);
BW = imfill(BW, 'holes');

[L, num] = bwlabel(BW);
stats = regionprops(L, 'BoundingBox', 'Area');

digits = {};
boxes = [];
labels_kept = [];

for i = 1:num
    if stats(i).Area >= cfg.minObjectArea
        boxes(end+1,:) = stats(i).BoundingBox; %#ok<AGROW>
        labels_kept(end+1) = i; %#ok<AGROW>
    end
end

% Trier les chiffres de gauche à droite
if ~isempty(boxes)
    [~, ord] = sort(boxes(:,1), 'ascend');
    labels_kept = labels_kept(ord);
end

for i = 1:length(labels_kept)
    mask = (L == labels_kept(i));

    % Rogner au plus juste
    [r, c] = find(mask);
    if isempty(r) || isempty(c)
        continue;
    end
    mask = mask(min(r):max(r), min(c):max(c));

    % Redimensionnement propre
    mask = imresize(mask, cfg.resizeDigitTo, 'nearest');
    mask = logical(mask);

    digits{end+1} = mask; %#ok<AGROW>
end

end

%% ============================================================
% FEATURES
%% ============================================================

function feat = extract_features(BW)

BW = logical(BW);
BW = bwareaopen(BW, 5);

[L, num] = bwlabel(BW);

if num == 0
    feat = zeros(1,5);
    return;
end

% garder la plus grande composante
stats0 = regionprops(L, 'Area');
areas = [stats0.Area];
[~, idxMax] = max(areas);
BW = (L == idxMax);

s = regionprops(BW, 'Area', 'Perimeter', 'Eccentricity', 'Extent', 'Solidity');

if isempty(s)
    feat = zeros(1,5);
    return;
end

feat = [
    s.Area, ...
    s.Perimeter, ...
    s.Eccentricity, ...
    s.Extent, ...
    s.Solidity ...
];

end

%% ============================================================
% UTILS
%% ============================================================

function I = rgb2gray_if_needed(I)
if ndims(I) == 3
    I = rgb2gray(I);
end
end

%% ============================================================
% EVALUATION
%% ============================================================

function acc = evaluate(cfg, model)

files = dir(fullfile(cfg.testDir, '*.png'));

if isempty(files)
    acc = 0;
    return;
end

correct = 0;
validCount = 0;

for i = 1:length(files)

    trueCode = regexp(files(i).name, '\d{5}', 'match');

    if isempty(trueCode)
        continue;
    end

    trueCode = trueCode{1};

    pred = recognize_postal_code(fullfile(files(i).folder, files(i).name), model, cfg);

    validCount = validCount + 1;

    if strcmp(pred, trueCode)
        correct = correct + 1;
    end
end

if validCount == 0
    acc = 0;
else
    acc = correct / validCount;
end

end