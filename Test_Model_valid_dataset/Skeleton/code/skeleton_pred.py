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
    
    batch_size = model.batch_size
    num_class = model.num_class
    data_size = data.data_size
    iterations = 32
    all_logits = np.zeros((iterations,batch_size,num_class),dtype=np.float32)
    

    start_time = time.time()

    all_correct_top_1 = 0
    all_correct_top_5 = 0
    total_loss = 0.0
    

    for iter in range(iterations):
        state = session.run(model.initial_state)
        fetches = [model.model_logits, model.correct_num_top_1, model.correct_num_top_5, model.cost, model.final_state, model.merged, eval_op] 

        feed_dict = {} 
        data_features,data_labels = data.next_batch(batch_size)
        feed_dict[model.data_features] = data_features
        feed_dict[model.labels] = data_labels

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c   
            feed_dict[h] = state[i].h

        all_logits[iter,:,:],correct_top_1, correct_top_5, loss, state, merged, _ = session.run(fetches, feed_dict)

        assert not np.isnan(loss),'loss = NaN'
        total_loss = total_loss + loss

        print ("accury(top 1): %.3f Test accury(top 5): %.3f Loss %.3f" % (correct_top_1 * 1.0 / batch_size, correct_top_5 * 1.0 / batch_size, loss) )
        

        all_correct_top_1 = correct_top_1 + all_correct_top_1
        all_correct_top_5 = correct_top_5 + all_correct_top_5

    save_logits =  np.max(all_logits,axis=0)
    np.save('logits.npy',save_logits)
    print ('logits saved!')
    print (save_logits.shape)
    
    return all_correct_top_1 * 1.0 / (iterations * batch_size) , all_correct_top_5 * 1.0 / (iterations * batch_size), total_loss * 1.0 / iterations
    
def run_train():
    
    test_config = lstmmodel.ModelConfig()
    #test_config.batch_size = 1
    test_config.batch_size = lstmmodel.TEST_DATA_SIZE  

    Session_config = tf.ConfigProto(allow_soft_placement = True)
    Session_config.gpu_options.allow_growth=True     

    with tf.Graph().as_default(), tf.Session(config=Session_config) as sess:
        with tf.device('/gpu:1'):
            initializer = tf.random_uniform_initializer(-test_config.init_scale, 
                                                        test_config.init_scale)
            with tf.variable_scope("model", reuse=None,initializer=initializer):
                model_test = lstmmodel.LSTMModel(test_config,'test',is_training=False)
            
                
            data_test = dataset.DataSet(FLAGS.file_path_test,FLAGS.data_root_dir,lstmmodel.TEST_DATA_SIZE,test_config.num_steps,test_config.feature_size,is_train_set=False)
            
           
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100)
            last_epoch = load_model(sess, saver,FLAGS.saveModelPath)
            print ('start: ',last_epoch + 1)
    
            
                
          
            test_accury_1,test_accury_5,test_loss = run_epoch(sess, model_test, data_test, tf.no_op(),2,test_writer,istraining=False) 
            print("Test accury(top 1): %.3f Test accury(top 5): %.3f Loss %.3f" % (test_accury_1,test_accury_5,test_loss  ))          
           
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
      default='../labels/fake_v_label_len',
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



