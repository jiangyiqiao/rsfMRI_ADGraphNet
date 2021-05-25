clc;
clear;
load('NEL/kalmancorr_0.01_0.9.mat');
rand("state", 0);
sample_num = size(corr, 1);
datas = corr(:,1);
labels = corr(:,4); 
index = (1:sample_num);
indexs = cell(sample_num,1);
for i=1:sample_num
    indexs{i} = index(i);
end
% ��������ռȫ�����ݵı���
K = 5;

% ���Լ�����
Indices = crossvalind('Kfold', sample_num, K);

%ѭ��5�Σ��ֱ�ȡ����i������Ϊ����������������������Ϊѵ������
for i = 1:K
    testIndices = (Indices == i);
    trainIndices = ~testIndices;
    % ѵ������ѵ����ǩ
    trainX = datas(trainIndices, :);
    trainY = labels(trainIndices, :);
    testX = datas(testIndices, :);
    testY = labels(testIndices, :);
    trainIndex = indexs(trainIndices, :);
    testIndex = indexs(testIndices, :);
    data = trainX;
    label = trainY;
    save(strcat('NEL/train/0.8/data_',num2str(i),'.mat'), 'data', 'label');
    data = testX;
    label = testY;
    save(strcat('NEL/val/0.8/data_',num2str(i),'.mat'), 'data', 'label');
end

% ���Լ��Ͳ��Ա�ǩ

