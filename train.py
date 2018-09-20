import tensorflow as tf
import argparse
import Experiments
import input_dataset
import time
slim = tf.contrib.slim
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--GPU', default='0', help='the index of gpu')
parser.add_argument('--checkpoint_dir', default='/data/zengdifei/TF_gaze/check_log', help='path of checkpoints')
parser.add_argument('--eyeball', default='left', help='the choose of eyeball')
parser.add_argument('--mission', default='theta', help='the mission of direction')
parser.add_argument('--batch_size', default=256, help='the size of examples in per batch')
parser.add_argument('--epoch', default=300, help='the train epoch')
parser.add_argument('--moving_average_decay', default=0.9999, help='moving average decay')
parser.add_argument('--num_epochs_per_decay', default=150, help='num epochs per decay')
parser.add_argument('--learning_rate_decay_factor', default=0.5, help='learning rate decay factor')
parser.add_argument('--learning_rate_initial', default=1e-2, help='learning rate initial')
parser.add_argument('--train_num', default=51484, help='the numbers of train dataset')
parser.add_argument('--data_type', default='M', help='the silent or move')
parser.add_argument('--data_dir', default='/data/zengdifei/TF_gaze/dataset/', help='the directory of training data')
parser.add_argument('--summary_dir', default='/data/zengdifei/TF_gaze/event_log', help='the directory of summary')
parser.add_argument('--net_name', default='InceptionV3', help='the name of the network')
parser.add_argument('--image_height', default=144, help='the height of image')
parser.add_argument('--image_width', default=240, help='the width of image')
args = parser.parse_args()
args.batch_size = int(args.batch_size)
args.num_epochs_per_decay = int(args.num_epochs_per_decay)
args.learning_rate_decay_factor = float(args.learning_rate_decay_factor)
args.data_dir = args.data_dir + args.data_type

if args.data_type == 'SM':
    if args.eyeball == 'left':
        args.train_num = 122923
    else:
        args.train_num = 116026
elif args.data_type == 'S':
    if args.eyeball == 'left':
        args.train_num = 71439
    else:
        args.train_num = 68636
elif args.data_type == 'M':
    if args.eyeball == 'left':
        args.train_num = 51484
    else:
        args.train_num = 47390
elif args.data_type == 'MPIIGaze':
    if args.eyeball == 'left':
        args.train_num = 156522
    else:
        args.train_num = 156522
elif args.data_type == 'MPIIIGaze':
    if args.eyeball == 'left':
        args.train_num = 93976
    else:
        args.train_num = 93872


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
    data_dir = args.data_dir + '/train_left.tfrecords'
    if args.mission == 'theta':
        half_classes = Experiments.half_classes_left_theta
        summary_dir = args.summary_dir + '/left_theta'
        checkpoint_dir = args.checkpoint_dir + '/left_theta/left_theta_model.ckpt'
        loss_name = 'loss_left_theta'
    else:
        half_classes = Experiments.half_classes_left_phi
        summary_dir = args.summary_dir + '/left_phi'
        checkpoint_dir = args.checkpoint_dir + '/left_phi/left_phi_model.ckpt'
        loss_name = 'loss_left_phi'
else:
    data_dir = args.data_dir + '/train_left.tfrecords'
    if args.mission == 'theta':
        half_classes = Experiments.half_classes_right_theta
        summary_dir = args.summary_dir + '/right_theta'
        checkpoint_dir = args.checkpoint_dir + '/right_theta/right_theta_model.ckpt'
        loss_name = 'loss_right_theta'
    else:
        half_classes = Experiments.half_classes_right_phi
        summary_dir = args.summary_dir + '/right_phi'
        checkpoint_dir = args.checkpoint_dir + '/right_phi/right_phi_model.ckpt'
        loss_name = 'loss_right_phi'

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

def arch_net(image, is_training, dropout_keep_prob, half_classes):
    arg_scope = net_arg_scope()
    with slim.arg_scope(arg_scope):
        if args.net_name == 'resnet_v2':
            logits, end_points = Net(image,
                                 num_classes=int(half_classes * 2 + 1), 
                                 is_training=is_training)
        else:
            logits, end_points = Net(image,
                                     num_classes=int(half_classes * 2 + 1), 
                                     is_training=is_training, 
                                     dropout_keep_prob=dropout_keep_prob, 
                                     create_aux_logits=False)
    return logits, end_points

def g_parameter(checkpoint_exclude_scopes):
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    print (exclusions)
    variables_to_restore = []
    variables_to_train = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                variables_to_train.append(var)
                print (var.op.name)
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore, variables_to_train

def train():

    print('Tensorflow begin')

    with tf.Graph().as_default():

        image, labels= input_dataset.train_input(data_dir, args.batch_size, half_classes, args.mission)

        labels = tf.reshape(labels, [args.batch_size])

        logits, end_points = arch_net(image, True, 0.5, half_classes)

        variables_to_restore,variables_to_train = g_parameter(args.net_name)

        # loss function
        
        if 'AuxLogits' in end_points:
            slim.losses.add_loss(tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, 
                                                                                       logits=end_points['AuxLogits'], 
                                                                                       weights=0.4)))
        slim.losses.add_loss(tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)))

        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar(loss_name, total_loss)

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)

        num_steps_per_epoch = int(args.train_num / args.batch_size)
        decay_steps = num_steps_per_epoch * args.num_epochs_per_decay

        lr = tf.train.exponential_decay(args.learning_rate_initial,
                                        global_step,
                                        decay_steps,
                                        args.learning_rate_decay_factor,
                                        staircase=True)
        
        tf.summary.scalar('learning_rate', lr)

        var_list = variables_to_train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, var_list=var_list, global_step = global_step)

        sess = tf.Session()
        init = tf.global_variables_initializer()

        net_vars = variables_to_train
        saver = tf.train.Saver(net_vars, max_to_keep = 0)
        merged = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=True)) as sess:

            init.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)

            step = 0
            for i in range(args.epoch):
                for j in range(num_steps_per_epoch):
                    step = step + 1
                    start_time = time.time()
                    loss_value, learning_rate_value, step, _ = sess.run([total_loss, lr, global_step, train_op])
                    duration = time.time() - start_time
        
                    if step % 10 == 0:
                        num_examples_per_step = args.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration

                        format_str = ('epoch %d step %d, loss = %.4f, lr = %.5f (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (i, step, loss_value, learning_rate_value, examples_per_sec, sec_per_batch))

                        summary = sess.run(fetches=merged)
                        summary_writer.add_summary(summary, step)

                    if step % 500 == 0:
                        saver.save(sess, checkpoint_dir, global_step=step)

            coord.request_stop()
            coord.join(threads)

        sess.close()

if __name__ == '__main__':

    print ("-----------------------------train.py start--------------------------")
    train()
