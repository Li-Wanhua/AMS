from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import dataset
import lstmmodel

# Basic model parameters
FLAGS = None
# 0 for train, 1 for valid
summary_global_step = [0,0,0]
lr_down_step = [60,80]

def load_model(sess, saver,ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)

    if latest_ckpt:
        print ('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1
    


def run_epoch(session, model, data, eval_op,step_index,summary_writer,istraining=True):
    """Runs the model on the given data."""
    batch_size = model.batch_size
    data_size = data.data_size
    iterations = data_size // batch_size
    if data_size % batch_size != 0:
        iterations = iterations + 1

    start_time = time.time()

    all_correct_top_1 = 0
    all_correct_top_5 = 0
    total_loss = 0.0
    state = session.run(model.initial_state)

    for iter in range(iterations):
        fetches = [model.correct_num_top_1, model.correct_num_top_5, model.cost, model.final_state, model.merged, eval_op] 

        feed_dict = {} 
        data_features,data_labels = data.next_batch(batch_size)
        feed_dict[model.data_features] = data_features
        feed_dict[model.labels] = data_labels

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c   
            feed_dict[h] = state[i].h

        correct_top_1, correct_top_5, loss, state, merged, _ = session.run(fetches, feed_dict)

        assert not np.isnan(loss),'loss = NaN'
        total_loss = total_loss + loss

        if step_index > -0.5:
            summary_writer.add_summary(merged, summary_global_step[step_index])
            summary_global_step[step_index] = summary_global_step[step_index] + 1

        all_correct_top_1 = correct_top_1 + all_correct_top_1
        all_correct_top_5 = correct_top_5 + all_correct_top_5
        '''
        if istraining:
            if iter % 200 == 1:
                print("%.3f: accury(top 1): %.3f accury(top 5): %.3f loss %.3f speed: %.1f sec" %
                  (iter * 1.0 / iterations, correct_top_1 * 1.0 / batch_size, correct_top_5 * 1.0 / batch_size, total_loss * 1.0 /(iter + 1),
                   (time.time() - start_time) * 1.0 / ((iter + 1) * model.batch_size) ))
        '''
        '''
        else:
            if iter % 200 == 1:
                print("%.3f: accury(top 1): %.3f accury(top 5): %.3f loss %.3lf speed: %.1f sec" %
                  (iter * 1.0 / iterations, correct_top_1 * 1.0 / batch_size, correct_top_5 * 1.0 / batch_size, total_loss * 1.0 /(iter + 1)
                   (time.time() - start_time) * 1.0 / (iter * model.batch_size) ))
        '''
    if istraining:
        return all_correct_top_1 * 1.0 / (iterations * batch_size) , all_correct_top_5 * 1.0 / (iterations * batch_size), total_loss * 1.0 / iterations
    else:
        return all_correct_top_1 * 1.0 / data_size, all_correct_top_5 * 1.0 / data_size, total_loss * 1.0 / iterations

def run_train():
    
    train_config = lstmmodel.ModelConfig()

    eval_config = lstmmodel.ModelConfig()
    #eval_config.batch_size = 1
    eval_config.batch_size = lstmmodel.VALID_DATA_SIZE

    test_config = lstmmodel.ModelConfig()
    #test_config.batch_size = 1
    test_config.batch_size = lstmmodel.TEST_DATA_SIZE  

    Session_config = tf.ConfigProto(allow_soft_placement = True)
    Session_config.gpu_options.allow_growth=True     

    with tf.Graph().as_default(), tf.Session(config=Session_config) as sess:
        with tf.device('/gpu:0'):
            initializer = tf.random_uniform_initializer(-train_config.init_scale, 
                                                        train_config.init_scale)
            with tf.variable_scope("model", reuse=None,initializer=initializer):
                model_train = lstmmodel.LSTMModel(train_config,'train')   
            with tf.variable_scope("model", reuse=True,initializer=initializer):
                model_valid = lstmmodel.LSTMModel(eval_config,'valid',is_training=False)
                model_test = lstmmodel.LSTMModel(test_config,'test',is_training=False)
            data_train = dataset.DataSet(FLAGS.file_path_train,FLAGS.data_root_dir,lstmmodel.TRAIN_DATA_SIZE,train_config.num_steps,train_config.feature_size)
            data_valid = dataset.DataSet(FLAGS.file_path_valid,FLAGS.data_root_dir,lstmmodel.VALID_DATA_SIZE,eval_config.num_steps,eval_config.feature_size,is_train_set=False)
            data_test = dataset.DataSet(FLAGS.file_path_test,FLAGS.data_root_dir,lstmmodel.TEST_DATA_SIZE,test_config.num_steps,test_config.feature_size,is_train_set=False)
            
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(max_to_keep=100)
            last_epoch = load_model(sess, saver,FLAGS.saveModelPath)
            print ('start: ',last_epoch + 1)
    
            for i in range(last_epoch + 1,train_config.max_epoch):
                new_learning_rate = train_config.learning_rate
                for  k in lr_down_step:
                    if i > k:
                        new_learning_rate = new_learning_rate / 10.0
                model_train.assign_lr(sess, new_learning_rate) 
    
                print("Epoch: %d Learning rate: %.3f" % (i, sess.run(model_train.lr)))
                
                train_accury_1,train_accury_5,train_loss = run_epoch(sess, model_train, data_train, model_train.train_op,0,train_writer) 
                print("Epoch: %d Train accury(top 1): %.3f Train accury(top 5): %.3f Loss %.3f" % (i, train_accury_1,train_accury_5,train_loss))
                
                saver.save(sess, FLAGS.saveModelPath + '/model.ckpt', global_step=i)

                #valid_accury_1,valid_accury_5,valid_loss = run_epoch(sess, model_valid, data_valid, tf.no_op(),1,valid_writer,istraining=False) 
                #print("Epoch: %d Valid accury(top 1): %.3f Valid accury(top 5): %.3f Loss: %.3f" % (i, valid_accury_1,valid_accury_5,valid_loss))
                
                test_accury_1,test_accury_5,test_loss = run_epoch(sess, model_test, data_test, tf.no_op(),2,test_writer,istraining=False) 
                print("Epoch: %d Test accury(top 1): %.3f Test accury(top 5): %.3f Loss %.3f" % (i,test_accury_1,test_accury_5,test_loss  ))          
            
            train_writer.close()
            valid_writer.close()
            test_writer.close()

            print("Training step is compeleted!")  
            
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.saveModelPath):
      tf.gfile.MakeDirs(FLAGS.saveModelPath)

  run_train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--file_path_train',
      type=str,
      default='../labels/train_label_len',
      help='file_path is the path of [video_path label] file.'
  )
  parser.add_argument(
      '--file_path_valid',
      type=str,
      default='../labels/valid_label_len',
      help='file_path is the path of [video_path label] file.'
  )
  parser.add_argument(
      '--file_path_test',
      type=str,
      default='../labels/valid_label_len',
      help='file_path is the path of [video_path label] file.'
  )
  parser.add_argument(
      '--data_root_dir',
      type=str,
      default='/home/share/chaLearn-Iso/skeleton_Full_Features/',
      help='data_root_dir is the root used for video_path,so we can use data_root_dir +  video_path to access video.'
  )
  parser.add_argument(
      '--saveModelPath',
      type=str,
      default='../ModelPara',
      help='Directory to put model parameter.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='../Modellog',
      help='Directory to put the log data.'
  )


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



