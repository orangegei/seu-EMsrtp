clear all;
close all;

my = xlsread('美国_外汇.xls');  % load data

asvar = {'volatility'; 'exrate'};    % variable names
nlag = 2;                   % # of lags

setvar('data', my, asvar, nlag); % set data

setvar('fastimp', 1);       % fast computing of response

mcmc(10000);                % MCMC

drawimp([4 8 12], 1);       % draw impulse reponse(1)
                            % : 4-,8-,12-period ahead
                            
drawimp([30 60 90], 0);		% draw impulse response(2)
                            % : response at t=30,60,90