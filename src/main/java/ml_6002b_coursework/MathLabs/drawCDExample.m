% creating a demsar diagram with gavins code
% File should contain ERRORS, not ACCURACIES. No headers
%addpath('C:\Users\ajb\Dropbox\Code\Matlab');
% Change to file name for errors

DATA_PATH='C:\UCI\UCIResults\UCI_HESC.csv';
names={'HESC','NN','NB','C45','SVML','SVMQ','RandF','RotF','BayesNet'}

data=csvread(DATA_PATH)
critdiff(data,names,0.05)