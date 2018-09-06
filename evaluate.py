import tensorflow as tf
import argparse
import data_process
import Experiments
import train
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
scale = 1
parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoint_dir', default='/data/zengdifei/TF_gaze/check_log', help='path of checkpoints')
parser.add_argument('--eyeball', default='left', help='the choose of eyeball')
parser.add_argument('--mission', default='theta', help='the mission of direction')
parser.add_argument('--batch_size', default=256, help='the size of examples in per batch')
parser.add_argument('--data_type', default='M', help='the silent or move')
parser.add_argument('--data_dir', default='/data/zengdifei/TF_gaze/Test/', help='the directory of testing data')
parser.add_argument('--net_name', default='InceptionV3', help='the name of the network')
parser.add_argument('--image_height', default=144, help='the height of image')
parser.add_argument('--image_width', default=240, help='the width of image')
parser.add_argument('--eval_interval_secs', default=5, help='the interval seconds')
parser.add_argument('--max_step', default=60000, help='max steps')
parser.add_argument('--test_count', default=12, help='test count in per checkpoints')
args = parser.parse_args()
args.batch_size = int(args.batch_size)
args.data_dir = args.data_dir + args.data_type

if args.data_type == 'SM':
    if args.eyeball == 'left':
        args.max_step = 144000
    else:
        args.max_step = 135500
elif args.data_type == 'S':
    if args.eyeball == 'left':
        args.max_step = 83500
    else:
        args.max_step = 80000
elif args.data_type == 'M':
    if args.eyeball == 'left':
        args.max_step = 60000
    else:
        args.max_step = 55500
elif args.data_type == 'MPIIGaze':
    if args.eyeball == 'left':
        args.max_step = 183000
        args.data_dir = args.data_dir + '/p14_Left/'
    else:
        args.max_step = 183000
        args.data_dir = args.data_dir + '/p14_Right/'

if args.net_name == 'InceptionV3':
    from inception_v3 import inception_v3_arg_scope as net_arg_scope
    from inception_v3 import inception_v3 as Net
elif args.net_name == 'vgg16':
    from vgg import vgg_arg_scope as net_arg_scope
    from vgg import vgg_16 as Net
elif args.net_name == 'resnet_v2':
    from resnet_v2 import resnet_arg_scope as net_arg_scope
    from resnet_v2 import resnet_v2_50 as Net
elif args.net_name == 'InceptionV4':
    from inception_v4 import inception_v4_arg_scope as net_arg_scope
    from inception_v4 import inception_v4 as Net
elif args.net_name == 'InceptionResnetV2':
    from inception_resnet_v2 import inception_resnet_v2_arg_scope as net_arg_scope
    from inception_resnet_v2 import inception_resnet_v2 as Net

if args.eyeball == 'left':
    file_input = open('/data/zengdifei/TF_gaze/Test_left_' + args.data_type + '.txt', 'r')
    if args.mission == 'theta':
        half_classes = Experiments.half_classes_left_theta
        f = open("../result/test_result_left_theta.txt", "a")
    else:
        half_classes = Experiments.half_classes_left_phi
        f = open("../result/test_result_left_phi.txt", "a")
else:
    file_input = open('/data/zengdifei/TF_gaze/Test_right_' + args.data_type + '.txt', 'r')
    if args.mission == 'theta':
        half_classes = Experiments.half_classes_right_theta
        f = open("../result/test_result_right_theta.txt", "a")
    else:
        half_classes = Experiments.half_classes_right_phi
        f = open("../result/test_result_right_phi.txt", "a")

def get_batch(batch_size, dataset):
    eye_batch = []
    label_batch = []
    for i in range(batch_size):
        imfile = (args.data_dir + dataset.img_list[dataset.index]).replace('\\', "/")
        success, eye = data_process._Get_From_pic(imfile, 4)
        label = np.zeros(1)
        if args.mission == 'theta':
            label[0] = dataset.gaze_theta_list[dataset.index]
        else:
            label[0] = dataset.gaze_phi_list[dataset.index]
        label[0] = ((label[0] + scale) * half_classes) * (1/scale)

        dataset.index += 1
        if dataset.index >= dataset.size:
            dataset.index -= dataset.size
        if success == 0:
            continue

        eye = eye.reshape(args.image_height, args.image_width, 3)
        label = label.reshape(1)

        eye_batch.append(eye)
        label_batch.append(label)
    return eye_batch, label_batch

def evaluate():

    eye_placeholder = tf.placeholder(tf.float32, [args.batch_size, args.image_height, args.image_width, 3])
    label_placeholder = tf.placeholder(tf.int64, [args.batch_size, 1])

    logits, _ = train.arch_net(eye_placeholder, False, 1, half_classes)

    error_mean = 0.5 * scale * (180 / float(half_classes)) * tf.reduce_mean(tf.abs(
			tf.cast(tf.argmax(logits, 1), tf.float32) -
			tf.cast(tf.reshape(label_placeholder, [args.batch_size]), tf.float32)), axis=[0])

    variables_to_restore, variables_to_train = train.g_parameter(args.net_name)
    saver = tf.train.Saver(var_list=variables_to_train)

    class DATA(object):
        pass

    EVAL = DATA()
    EVAL.img_list = []
    #EVAL.pose_theta_list = []
    #EVAL.pose_phi_list = []
    EVAL.gaze_theta_list = []
    EVAL.gaze_phi_list = []
    EVAL.index = 0
    EVAL.size = 0

    for line in file_input:
        content = line.strip().split(' ')
        if float(content[1]) > -0.4 and float(content[1]) < 0.5 and float(content[2]) > -0.4 and float(content[2]) < 0.5:
            EVAL.img_list.append(content[0])
            #EVAL.pose_theta_list.append(float(content[1]))
            #EVAL.pose_phi_list.append(float(content[2]))
            EVAL.gaze_theta_list.append(float(content[1]))
            EVAL.gaze_phi_list.append(float(content[2]))
            EVAL.size += 1
        del content
    file_input.close()

    Min_Error = 65536; Best_step = 0; step = 500; flag = 0;

    while step < args.max_step:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('/data/zengdifei/TF_gaze/check_log/' + args.eyeball + '_' + args.mission + '/')
            if ckpt and ckpt.model_checkpoint_path:

                if args.eyeball == 'left':
                    if args.mission == 'theta':
                        f = open("../result/test_result_left_theta.txt", "a")
                    else:
                        f = open("../result/test_result_left_phi.txt", "a")
                else:
                    if args.mission == 'theta':
                        f = open("../result/test_result_right_theta.txt", "a")
                    else:
                        f = open("../result/test_result_right_phi.txt", "a")

                ckpt_dir = ckpt.model_checkpoint_path
                if step > int(ckpt_dir.split('/')[-1].split('-')[-1]):
                    time.sleep(1)
                    continue
                read_dir = ckpt_dir.split('-')[0] + '-' + str(step)
                saver.restore(sess, read_dir)
                eye_batch, label_batch = get_batch(args.batch_size, EVAL)
                error = sess.run(error_mean, feed_dict= {eye_placeholder: eye_batch,
														label_placeholder: label_batch})

                if error < Min_Error:
                    Min_Error = error; Best_step = step

                print('STEP = %d MAE = %.6f Best_Step = %d' % (step, error, Best_step), file=f)
                print('STEP = %d MAE = %.6f Best_Step = %d' % (step, error, Best_step))
                f.close()
            else:
                print('No checkpoint file found')
                continue

            flag = flag + 1
            if flag == args.test_count:
                step = step + 500
                flag = 0

def main():
    evaluate()

if __name__ == '__main__':
    main()
