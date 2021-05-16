import model
import data_create
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow SRCNN Example')

    parser.add_argument('--train_height', type = int, default = 33, help = "Train data size(height)")
    parser.add_argument('--train_width', type = int, default = 33, help = "Train data size(width)")
    parser.add_argument('--test_height', type = int, default = 700, help = "Test data size(height)")
    parser.add_argument('--test_width', type = int, default = 700, help = "Test data size(width)")
    parser.add_argument('--train_dataset_num', type = int, default = 10000, help = "Number of train datasets to generate")
    parser.add_argument('--test_dataset_num', type = int, default = 5, help = "Number of test datasets to generate")
    parser.add_argument('--train_cut_num', type = int, default = 10, help = "Number of train data to be generated from a single image")
    parser.add_argument('--test_cut_num', type = int, default = 1, help = "Number of test data to be generated from a single image")
    parser.add_argument('--train_path', type = str, default = "../../dataset/DIV2K_train_HR", help = "The path containing the train image")
    parser.add_argument('--test_path', type = str, default = "../../dataset/DIV2K_valid_HR", help = "The path containing the test image")
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help = "Learning_rate")
    parser.add_argument('--BATCH_SIZE', type = int, default = 32, help = "Training batch size")
    parser.add_argument('--EPOCHS', type = int, default = 500, help = "Number of epochs to train for")
   
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    parser.add_argument('--mode', type=str, default='train_model', help='train_datacreate, test_datacreate, train_model, evaluate')

    args = parser.parse_args()

    if args.mode == 'train_datacreate': #学習用データセットの生成
        datacreate = data_create.datacreate()
        train_x, train_y = datacreate.datacreate(args.train_path,       #切り取る動画のpath
                                            args.train_dataset_num,     #データセットの生成数
                                            args.train_cut_num,         #1枚の画像から生成するデータの数
                                            args.train_height,          #保存サイズ
                                            args.train_width)   
        path = "train_data_list"
        np.savez(path, train_x, train_y)

    elif args.mode == 'test_datacreate': #評価用データセットの生成
        datacreate = data_create.datacreate()
        test_x, test_y = datacreate.datacreate(args.test_path,
                                            args.test_dataset_num,
                                            args.test_cut_num,
                                            args.test_height,
                                            args.test_width)

        path = "test_data_list"
        np.savez(path, test_x, test_y)

    elif args.mode == "train_model": #学習
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")

        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.SRCNN()

        optimizers = tf.keras.optimizers.Adam(lr = args.learning_rate)
        train_model.compile(loss = "mean_squared_error",
                        optimizer = optimizers,
                        metrics = [psnr])

        train_model.fit(train_x,
                        train_y,
                        epochs = args.EPOCHS,
                        verbose = 2,
                        batch_size = args.BATCH_SIZE)

        train_model.save("SRCNN_model.h5")

    elif args.mode == "evaluate": #評価
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")

        result_path = "result"
        os.makedirs(result_path, exist_ok = True)

        npz = np.load("test_data_list.npz", allow_pickle = True)

        test_x = npz["arr_0"]
        test_y = npz["arr_1"]

        test_x = tf.convert_to_tensor(test_x, np.float32)
        test_y = tf.convert_to_tensor(test_y, np.float32)

        test_x /= 255
        test_y /= 255
            
        path = "SRCNN_model.h5"

        if os.path.exists(path):
            model = tf.keras.models.load_model(path, custom_objects={'psnr':psnr})
            pred = model.predict(test_x, batch_size = 1)

            for p in range(len(test_y)):
                pred[p][pred[p] > 1] = 1
                pred[p][pred[p] < 0] = 0
                ps_pred = psnr(tf.reshape(test_y[p], [args.test_height, args.test_width, 1]), pred[p])
                ps_bicubic = psnr(tf.reshape(test_y[p], [args.test_height, args.test_width, 1]), tf.reshape(test_x[p], [args.test_height, args.test_width, 1]))

                low_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_x[p] * 255, [args.test_height, args.test_width]))
                cv2.imwrite(result_path + "/" + str(p) + "_low" + ".jpg", low_img)     #LR
                high_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_y[p] * 255, [args.test_height, args.test_width]))
                cv2.imwrite(result_path + "/" + str(p) + "_high" + ".jpg", high_img)   #HR
                pred_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(pred[p] * 255, [args.test_height, args.test_width]))
                cv2.imwrite(result_path + "/" + str(p) + "_pred" + ".jpg", pred_img)   #pred

                print("num:{}".format(p))
                print("psnr_pred:{}".format(ps_pred))
                print("psnr_bicubic:{}".format(ps_bicubic))

