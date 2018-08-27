import tensorflow as tf
import numpy as np
import cv2 as cv
import os

def _Get_From_pic(faces_path, scale_ratio):
    img = cv.imread(faces_path)
    height, width = img.shape[:2]
    img = cv.resize(img, (scale_ratio * width, scale_ratio * height), interpolation = cv.INTER_CUBIC)

    return 1, img

def main():

    file_input = open('/media/zengdifei/Passport/MPIIGaze/Normalization/Left/Train_left.txt', 'r')
    img_list = []
    #pose_x_list = []
    #pose_y_list = []
    gaze_x_list = []
    gaze_y_list = []
    for line in file_input:
        content = line.strip().split(' ')
        img_list.append(content[0])
        #pose_x_list.append(float(content[1]))
        #pose_y_list.append(float(content[2]))
        gaze_x_list.append(float(content[1]))
        gaze_y_list.append(float(content[2]))
        del content
    file_input.close()

    writer = tf.python_io.TFRecordWriter('/data/zengdifei/TF_gaze/dataset_new/train_left.tfrecords')

    for in_idx, in_ in enumerate(img_list):
        temp = in_.split('_')
        imfile = '/media/zengdifei/Passport/MPIIGaze/Normalization/Left/' + temp[0] + '/' + in_
        success, img = _Get_From_pic(imfile, 4)

        if success == 0:
            continue;

        img_raw = img.tobytes()

        #pose = np.array(np.zeros(2))
        #pose[0] = pose_x_list[in_idx]
        #pose[1] = pose_y_list[in_idx]
        #pose_raw = pose.tobytes()

        label = np.array(np.zeros(2))
        label[0] = gaze_x_list[in_idx]
        label[1] = gaze_y_list[in_idx]
        label_raw = label.tobytes()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value = [img_raw])),
            #'pose_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value = [pose_raw])),
            'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value = [label_raw]))
        }))

        print('data for train: {} / {}'.format(in_idx, len(img_list)))

        writer.write(example.SerializeToString())

        del temp

    writer.close()

    file_input = open('/media/zengdifei/Passport/MPIIGaze/Normalization/Right/Train_right.txt', 'r')
    img_list = []
    #pose_x_list = []
    #pose_y_list = []
    gaze_x_list = []
    gaze_y_list = []
    for line in file_input:
        content = line.strip().split(' ')
        img_list.append(content[0])
        #pose_x_list.append(float(content[1]))
        #pose_y_list.append(float(content[2]))
        gaze_x_list.append(float(content[1]))
        gaze_y_list.append(float(content[2]))
        del content
    file_input.close()

    writer = tf.python_io.TFRecordWriter('/data/zengdifei/TF_gaze/dataset_new/train_right.tfrecords')

    for in_idx, in_ in enumerate(img_list):
        temp = in_.split('_')
        imfile = '/media/zengdifei/Passport/MPIIGaze/Normalization/Right/' + temp[0] + '/' + in_
        success, img = _Get_From_pic(imfile, 4)

        if success == 0:
            continue;

        img_raw = img.tobytes()

        #pose = np.array(np.zeros(2))
        #pose[0] = pose_x_list[in_idx]
        #pose[1] = pose_y_list[in_idx]
        #pose_raw = pose.tobytes()

        label = np.array(np.zeros(2))
        label[0] = gaze_x_list[in_idx]
        label[1] = gaze_y_list[in_idx]
        label_raw = label.tobytes()

        example = tf.train.Example(features = tf.train.Features(feature = {
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value = [img_raw])),
            #'pose_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value = [pose_raw])),
            'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value = [label_raw]))
        }))

        print('data for train: {} / {}'.format(in_idx, len(img_list)))

        writer.write(example.SerializeToString())

        del temp

    writer.close()

if __name__ == '__main__':
    main()
