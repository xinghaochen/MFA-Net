close all; clear all; clc;

data_dir = '/home/workspace/Datasets/DHG2016';
gesture = 5;
finger = 1;
subject = 2;
essai = 1;
for idx = 1:135
    depth_path = sprintf('%s/gesture_%d/finger_%d/subject_%d/essai_%d/depth_%d.png', data_dir, gesture, finger, subject, essai, idx); 
    depth = imread(depth_path);
    imshow(uint8(depth/4));
    pause(0.01)
end