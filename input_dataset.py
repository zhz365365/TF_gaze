import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from PIL import Image

scale = 1

def read_image(filename_queue):

    class GazeRecord(object):
        pass

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           #'pose_raw': tf.FixedLenFeature([], tf.string),
                                           'label_raw': tf.FixedLenFeature([], tf.string)
                                       })

    image = GazeRecord()

    image.eye = tf.decode_raw(features['img_raw'], tf.uint8)
    image.eye = tf.reshape(image.eye, [144, 240, 3])

    #image.pose = tf.decode_raw(features['pose_raw'], tf.float64)
    #image.pose = tf.reshape(image.pose, [2])
    
    image.label = tf.decode_raw(features['label_raw'], tf.float64)
    image.label = tf.reshape(image.label, [2])

    return image

def _generate_image_and_label_batch(eye, label, min_queue_examples, batch_size, shuffle):

    num_preprocess_threads = 256

    if shuffle:
        eye_batch, label_batch = tf.train.shuffle_batch(
            [eye, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        eye_batch, label_batch = tf.train.batch(
            [eye, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size)

    return eye_batch, label_batch

def train_input(data_dir, batch_size, half_classes, mission):

    filename = [data_dir]

    filename_queue = tf.train.string_input_producer(filename)

    image = read_image(filename_queue)

    eye = tf.cast(image.eye, tf.float32)

    label = tf.cast(image.label, tf.float32)
    if mission == 'theta':
        new_label = label[0]
    else:
        new_label = label[1]
    new_label = tf.reshape(new_label, [1])
    new_label = (new_label + tf.constant(value=scale, dtype=tf.float32, shape=[1])) * (1/scale)
    new_label = tf.to_int64(tf.round(new_label * half_classes))

    return _generate_image_and_label_batch(eye, new_label, 64, batch_size, shuffle = True)

def main():
    with tf.Graph().as_default():
        filename = ['/data/zengdifei/TF_gaze/dataset/M/train_left.tfrecords']

        filename_queue = tf.train.string_input_producer(filename)

        image = read_image(filename_queue)

        eye = image.eye
        label = tf.cast(image.label, tf.float32)
        new_label = label[0]
        new_label = tf.reshape(new_label, [1])
        new_label = new_label + tf.constant(value=1, dtype=tf.float32, shape=[1])
        new_label = tf.to_int64(tf.round(new_label * Experiments.half_classes))

        with tf.Session(config = tf.ConfigProto(log_device_placement = False)) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(1024):
                Eye, Label, old_label = sess.run([eye, new_label, label])
                Eye = Eye[:,:,::-1]
                Eye = Image.fromarray(Eye, 'RGB')
                Eye.save('test_tfrecords/' + str(i) + '_face_' 
                                           + str(Label) + '_' 
                                           + str(old_label[0]) + '_'
                                           + str(old_label[1]) + '.jpg')

                print(old_label)

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()


















