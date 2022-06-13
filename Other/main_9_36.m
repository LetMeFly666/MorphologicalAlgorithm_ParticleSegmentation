clc;clear;
 

img=imread('9.36.jpg');
img = rgb2gray(img);
[rows, cols]=size(img);
 
% ��ֵ��
img=imbinarize(img);
 
% ������
se=strel('disk',3');  % Բ���ͽṹԪ��


img=imopen(img,se);  % ������
subplot(141);imshow(img);
title('��ֵ��','fontname','Microsoft YaHei UI Light','FontSize',16);
% LΪ��Ǿ���nΪ�ҵ����ӷ���������
[L,n]=bwlabel(img);
% ����߽��ص��Ŀ���
mask_bounds=img;
for k=1:rows
    if(L(k,1)~=0)
        I= L==L(k,1);
        mask_bounds(I)=0;
        I=0;
    end
end
for k=1:rows
    if(L(k,cols)~=0)
        I= L==L(k,cols);
        mask_bounds(I)=0;
        I=0;
    end
end
for k=1:cols
    if(L(1,k)~=0)
        [I]=find(L==L(1,k));
        mask_bounds(I)=0;
        I=0;
    end
end
for k=1:cols
    if(L(rows,k)~=0)
        [I]=find(L==L(rows,k));
        mask_bounds(I)=0;
        I=0;
    end
end
mask_bounds=img-mask_bounds;
subplot(142);imshow(mask_bounds);
title('��߽��ص�','fontname','Microsoft YaHei UI Light','FontSize',16);
% Particles overlapped with each other only, with an area of 300~450
f2=img;
for k=1:n
    [r,c]=find(L==k);
    % if( size(r,1)>300 && size(r,1)<450 )
    if (size(r, 1) >= 390)
        for i=1:size(r,1)
            f2(r(i),c(i))=0;
        end
    end 
end
f2=img-f2;
subplot(143);imshow(f2);
title('�˴��ص�','fontname','Microsoft YaHei UI Light','FontSize',16);
% No overlapping particles
f3=img;
for k=1:n
    [r,c]=find(L==k);
    if( size(r,1)<390 )
        for i=1:size(r,1)
            f3(r(i),c(i))=0;
        end
    end 
end
f3=img-f3;
f3=f3-mask_bounds;  % Exclude particles that overlap boundaries
subplot(144);imshow(f3);
title('���ص�','fontname','Microsoft YaHei UI Light','FontSize',16);
