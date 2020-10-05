import cv2
import os
import random
import glob
import numpy as np
import tensorflow as tf

#任意のフレーム数を切り出すプログラム
def save_frame(path,        #データが入っているファイルのパス
               data_number, #1枚の画像から切り取る写真の数
               cut_height,  #保存サイズ(縦)
               cut_width,   #保存サイズ(横)
               mag,         #縮小倍率
               ext='jpg'):

    #データセットのリストを生成
    low_data_list = []
    high_data_list = []

    path = path + "/*"
    files = glob.glob(path)
    
    for img in files:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape

        if cut_height > H or cut_width > W:
            return

        for q in range(data_number):
            ram_h = random.randint(0, H - cut_height)
            ram_w = random.randint(0, W - cut_width)
        
            cut_img = img[ram_h : ram_h + cut_height, ram_w: ram_w + cut_width]
            
            #ガウシアンフィルタでぼかしを入れる
            img1 = cv2.GaussianBlur(cut_img, (5, 5), 0)

            high_data_list.append(cut_img)
            low_data_list.append(img1)
    
    #numpy → tensor　+ 正規化
    low_data_list = tf.convert_to_tensor(low_data_list, np.float32)
    high_data_list = tf.convert_to_tensor(high_data_list, np.float32)
    low_data_list /= 255
    high_data_list /= 255

    return low_data_list, high_data_list

