import LmyGhostCorrection.*;

load ./myTest/myData/ksp-1.mat; % 载入 kspAllU kspPreScan preScanOption
kspAllU = permute(kspAllU, [2,3,1,4]);
ksp = permute(kspAllUInitial, [2,3,1,4]);
kspCor = [];
paraCor = [];
%%
for slice = 1:1:size(ksp,4)
    [kspC, paraC] = LmyGhostCorrection.oneDimLinearCorr_entropy(ksp(:, :, :, slice), 2);
%     [kspC, paraC] = LmyGhostCorrection.oneDimLinearCorr_entropy(kspC, 2);
    kspCor(:, :, :, slice) = kspC;
    paraCor(:, :, :, slice) = paraC; 
end

%% 显示预扫描给出的偏差
p = fftshift(ifft(kspPreScan,[],2),2);
deltaP = squeeze(p(1, :, 1:2:end, :).*conj(p(1, :, 2:2:end, :)));
figure(5);
plot(angle(deltaP(:, 1, 2)));
title("预扫描校正结果");

%% 显示结果
imCorByPreScan = fftshift(ifft2c3(kspAllU,256,256),1);
imInitial = fftshift(ifft2c3(ksp,256,256),2);
imCorByLPC = fftshift(ifft2c3(kspCor,256,256),2);

imInitial = squeeze(sqrt(sum(abs(imInitial).^2, 3)));
imCorByLPC = squeeze(sqrt(sum(abs(imCorByLPC).^2, 3)));
imCorByPreScan = squeeze(sqrt(sum(abs(imCorByPreScan).^2, 3)));
figure(1);
imshow3(imInitial, [], [3,ceil(size(imCorByPreScan, 3)/3)]);
title("initial");
figure(2);
imshow3(imCorByLPC, [], [3,ceil(size(imCorByPreScan, 3)/3)]);
title("LPC");
figure(3);
imshow3(imCorByPreScan, [], [3,ceil(size(imCorByPreScan, 3)/3)]);
title("preScan");


%% 进行 SAKE 测试

% clear all; close all;clc

import LmyGhostCorrection.*;
% load ./myTest/myData/ksp-1.mat; % 载入 kspAllU kspPreScan preScanOption
% nSeg=preScanOption.shotNum*2;
% 填零补全矩阵
nSeg = 4;
kspAllU = permute(kspCor, [3,1,2,4]);
kspAll = zeros(size(kspAllU,1), size(kspAllU,2),size(kspAllU,3)+30,size(kspAllU,4));
kspAll(:, :, 31:end, :) = kspAllU;
kspFull = [];
for i = 1:1:3 
    [~, kt] = pocs(kspAll(:, :, :, i), 100, true);
    kspFull(:, :, :,i) = kt;
end

epi_kxkyzc_2shot_lpcCor = permute(kspFull, [2,3,4,1]);

nSlice=size(epi_kxkyzc_2shot_lpcCor,3);
nCoil=size(epi_kxkyzc_2shot_lpcCor,4);
%%
ncalib = 90;
% threshold_list=[4*ones(1,10),4.5*ones(1,15),5*ones(1,30),4.5*ones(1,15),4*ones(1,10)]; % good for phantom
% threshold_list=[linspace(4,5,40),linspace(5,4,40)];
threshold_list=[linspace(3,4,5),linspace(4,4,15)];
ksize = [3,3]; % ESPIRiT kernel-window-size
sakeIter = 50;
% wnthresh = 4.5; % 3 or 4 good for brain

epi_kxkyzc_2shot_sakeCor=zeros(ncalib,ncalib,nSlice,nCoil);
epi_kxkyzc_2shot_sakeCor_fullCoils=...
    cat(4,epi_kxkyzc_2shot_sakeCor,epi_kxkyzc_2shot_sakeCor,epi_kxkyzc_2shot_sakeCor,epi_kxkyzc_2shot_sakeCor);
calib_sake_all = [];
for iSlice=1:nSlice
    % for iSlice=5
    disp(iSlice)
    DATA=squeeze(epi_kxkyzc_2shot_lpcCor(:,:,iSlice,:));
    
    % convert shot to VCC
    DATA_org=DATA;
    for iSeg=2:nSeg
        DATA=cat(3,DATA,circshift(DATA_org,-iSeg+1,2));
    end
    if exist('threshold_list','var')
        wnthresh = threshold_list(iSlice); % Window-normalized number of singular values to threshold
    end
    [sx,sy,Nc] = size(DATA);
    mask = zeros(size(DATA,1),size(DATA,2));
    mask(:,1:nSeg:end) = 1;
    DATA2recon=DATA.* repmat(mask,[1,1,size(DATA,3)]);
    DATAc = DATA;
    calibc = crop(DATAc,[ncalib,ncalib,size(DATA,3)]); %取中心的数据
    
    %% Perform SAKE reconstruction to recover the calibration area
    im = ifft2c(DATAc);
    disp('Performing SAKE recovery of calibration');
    tic; calib_sake = SAKEwithInitialValue(calibc, [ksize], wnthresh,sakeIter, 0,repmat(crop(mask,[ncalib,ncalib]),[1 1 size(DATA,3)]));toc
    calib_sake(:,:,end/4+1:end)=circshift(calib_sake(:,:,end/4+1:end),[0 1]);
    calib_sake(:,:,end/2+1:end)=circshift(calib_sake(:,:,end/2+1:end),[0 1]);
    calib_sake(:,:,end*3/4+1:end)=circshift(calib_sake(:,:,end*3/4+1:end),[0 1]);
    epi_kxkyzc_2shot_sakeCor_fullCoils(:,:,iSlice,:)= calib_sake;
    a=LmyGhostCorrection.pos_neg_add(ifft2c(calib_sake(:,:,1:end/4)),ifft2c(calib_sake(:,:,end/4+1:end/2)));
    b=LmyGhostCorrection.pos_neg_add(ifft2c(calib_sake(:,:,end/2+1:end*3/4)),ifft2c(calib_sake(:,:,end*3/4+1:end)));
    epi_kxkyzc_2shot_sakeCor(:,:,iSlice,:)=fft2c(LmyGhostCorrection.pos_neg_add(a,b))/4;
    calib_sake_all(:, :, :, iSlice) = calib_sake;
end

%% 显示结果
kspInitial = crop(epi_kxkyzc_2shot_lpcCor,[ncalib,ncalib,size(epi_kxkyzc_2shot_lpcCor,3),size(epi_kxkyzc_2shot_lpcCor,4)]); %取中心的数据
imInitial = ifft2c3(kspInitial, 128, 128);
imInitial = squeeze(sqrt(sum(abs(imInitial).^2, 4)));
figure();
imshow3(abs(imInitial), [], [1,3]);
title("initial image of center ksp");
im = (ifft2c3(epi_kxkyzc_2shot_sakeCor, 128,128));
im = sqrt(sum(abs(im).^2, 4));
figure();
imshow3(abs(im), [], [1,3]);