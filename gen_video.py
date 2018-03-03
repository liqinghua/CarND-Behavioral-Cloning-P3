#!/usr/bin/python

import cv2
import os
import glob


def imgs2video(imgs_dir, save_name):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (320, 160))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.jpg'))
    #print(imgs);

    for i in range(len(imgs)):
        imgname = imgs[i];
        frame = cv2.imread(imgname)
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    imgs2video('run2/', "output2.mp4")
