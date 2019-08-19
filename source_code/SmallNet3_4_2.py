import tensorflow as tf
import numpy as np
from PIL import Image

import os
import sys

#Error 출력 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#데이터셋 경로
PATH = "./"
PATH_DATASET = PATH + os.sep + "dataset"

#모델 저장 경로
MODEL_PATH = PATH + os.sep + "model"

FILE_TRAIN = PATH_DATASET + os.sep + "train.tfrecords"
FILE_TEST = PATH_DATASET + os.sep + "test.tfrecords"

tf.logging.set_verbosity(tf.logging.INFO)

#shape 출력
def print_activations(tensor):
    print(tensor.op.name, ' ', tensor.get_shape().as_list())

def input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })
        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.decode_raw(features['label'], tf.uint8)
        
        label = tf.cast(label,tf.int64) 
        
        image = tf.cast(image, tf.float32)
        return dict({'image':image}),label

    dataset = tf.data.TFRecordDataset(file_path).map(decode)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(256)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_label = iterator.get_next()
    return batch_features, batch_label

def my_model_fn(
    features, 
    labels,  
    mode):  

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    feature_columns = [tf.feature_column.numeric_column(key='image',shape=(256*256*3))]

    # Create the layer of input - 입력 레이어
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    
    input_layer = tf.reshape(input_layer,[-1,256,256,3])

    # HPF
    F0 = np.array([[[-1,-1,-1], [2,2,2], [-2,-2,-2], [2,2,2], [-1,-1,-1]],
                [[2,2,2], [-6,-6,-6], [8,8,8], [-6,-6,-6], [2,2,2]],
                [[-2,-2,-2], [8,8,8], [-12,-12,-12], [8,8,8], [-2,-2,-2]],
                [[2,2,2], [-6,-6,-6], [8,8,8], [-6,-6,-6], [2,2,2]],
                [[-1,-1,-1], [2,2,2], [-2,-2,-2], [2,2,2], [-1,-1,-1]]], dtype=np.float32)
    F1 = F0 / 12.

    Filter = np.array(F1, dtype=np.float32)
    print(Filter.shape)
    my_filter = tf.constant_initializer(value=Filter, dtype=tf.float32)

    HPF = tf.layers.Conv2D(filters = 1, activation = None, kernel_size= 5,trainable=False,strides=1,padding="SAME",use_bias = False,kernel_initializer=my_filter)(input_layer)
    print(HPF.shape)

    conv1 = tf.layers.conv2d(HPF, 64, 3, strides=2,padding='SAME', activation=tf.nn.relu)
    print(conv1.shape)

    conv2 = tf.layers.conv2d(conv1, 64, 3 , strides= 2,padding='SAME', activation=tf.nn.relu)
    print(conv2.shape)

    conv3 = tf.layers.conv2d(conv2, 128, 3 , strides= 2,padding='SAME', activation=tf.nn.relu)
    print(conv2.shape)

    conv4 = tf.layers.conv2d(conv3, 32, 3, strides = 1 ,activation=tf.nn.relu)
    print(conv3.shape)
    pool = tf.layers.max_pooling2d(conv4, pool_size= 3, strides = 2, padding='SAME')
   
    flatten = tf.layers.flatten(pool)
    print(pool.shape)

    fc1 = tf.layers.Dense(256, activation=tf.nn.relu)(flatten)
    drop1 = tf.layers.dropout(fc1, rate=0.5, name='drop1')

    fc2 = tf.layers.Dense(4096, activation=tf.nn.relu)(drop1)
    drop2 = tf.layers.dropout(fc2, rate=0.5, name='drop2')

    logits = tf.layers.Dense(12)(drop2)

    predictions = { 'class_ids': tf.argmax(input=logits, axis=1) }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={'my_accuracy': accuracy})

    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss,global_step=tf.train.get_global_step())
    
    tf.summary.scalar('my_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)


tf.logging.info("Before classifier construction")
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=MODEL_PATH,
    config = tf.estimator.RunConfig(save_checkpoints_steps=10000,keep_checkpoint_max=500) )  # Path to where checkpoints etc are stored
tf.logging.info("...done constructing classifier")


view = []
for i in range(0,262441,13122):
    checkpointPath = MODEL_PATH + os.sep + "model.ckpt-" + str(i)
    accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(FILE_TEST, False, 1),checkpoint_path=checkpointPath)["my_accuracy"]
    view.append(accuracy_score)

for index, a in enumerate(view):
    print("{0:d} Test Accuracy : {1:f}\n".format((index),a))

