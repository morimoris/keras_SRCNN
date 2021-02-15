import model
import data_create
import argparse
import os
import cv2

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    train_height = 33
    train_width = 33
    test_height = 700
    test_width = 700

    mag = 3.0
    cut_traindata_num = 10
    cut_testdata_num = 1

    train_file_path = "./train_data"
    test_file_path = "./test_data"

    BATSH_SIZE = 240
    EPOCHS = 1000
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='srcnn', help='srcnn, evaluate')

    args = parser.parse_args()

    if args.mode == "srcnn":
        train_x, train_y = data_create.save_frame(train_file_path,   #切り取る画像が入ったpath
                                                cut_traindata_num,  #データセットの生成数
                                                train_height, #保存サイズ
                                                train_width,
                                                mag)   #倍率
                                                
        model = model.SRCNN() 
        model.compile(loss = "mean_squared_error",
                        optimizer = opt,
                        metrics = [psnr])
#https://keras.io/ja/getting-started/faq/
        model.fit(train_x,
                    train_y,
                    epochs = EPOCHS)

        model.save("srcnn_model.h5")

    elif args.mode == "evaluate":
        path = "srcnn_model"
        exp = ".h5"
        new_model = tf.keras.models.load_model(path + exp, custom_objects={'psnr':psnr})

        new_model.summary()

        test_x, test_y = data_create.save_frame(test_file_path,   #切り取る画像が入ったpath
                                                cut_testdata_num,  #データセットの生成数
                                                test_height, #保存サイズ
                                                test_width,
                                                mag)   #倍率

        pred = new_model.predict(test_x)
        path = "result"
        os.makedirs(path, exist_ok = True)
        path = path + "/"

        ps = psnr(tf.reshape(test_y[0], [test_height, test_width, 1]), pred[0])
        print("psnr:{}".format(ps))

        before_res = tf.keras.preprocessing.image.array_to_img(tf.reshape(test_x[0], [test_height, test_width, 1]))
        change_res = tf.keras.preprocessing.image.array_to_img(tf.reshape(test_y[0], [test_height, test_width, 1]))
        y_pred = tf.keras.preprocessing.image.array_to_img(pred[0])

        before_res.save(path + "low_" + str(0) + ".jpg")
        change_res.save(path + "high_" + str(0) + ".jpg")
        y_pred.save(path + "pred_" + str(0) + ".jpg")

    else:
        raise Exception("Unknow --mode")
 
   
