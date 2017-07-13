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

import vgg16
import dataset

# Basic model parameters
FLAGS = None
TRAIN_SIZE = 35875
VALID_SIZE = 5784
TEST_SIZE = 5784
base_lr = 0.001
down_step = [11,14,17]
summary_global_step = [0,0,0]
short_term_loss = [6.9]
short_term_len = 100

class ModelConfig(object):
    init_scale = 0.01
    learning_rate = 1.0
    max_grad_norm = 5
    keep_epoch = 100
    max_epoch = 24
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 32
    num_class = 249

def dense_to_one_hot(labels_dense,num_class):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_class
    labels_one_hot = np.zeros((num_labels, num_class),dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1.0
    return labels_one_hot 

def load_model(sess, saver,ckpt_path,train_model):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)

    if latest_ckpt:
        print ('resume from', latest_ckpt)
        
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print ('building model from scratch')
        sess.run(tf.global_variables_initializer())
        train_model.load_pretrain_model(sess)
        return -1
    


def run_epoch(real_lr,session, keep_prob,log_file,batch_size, model, data, eval_op,step_index,summary_writer,istraining=True):
    """Runs the model on the given data."""
    
    data_size = data.data_size
    iterations = data_size // batch_size
    if data_size % batch_size != 0:
        iterations = iterations + 1

    start_time = time.time()

    all_correct_top_1 = 0
    all_correct_top_5 = 0
    total_loss = 0.0
    

    for iter in range(iterations):
        fetches = [model.learn_rate,model.correct_num_top_1, model.correct_num_top_5, model.loss,eval_op] 

        feed_dict = {} 
        data_features,data_labels = data.next_batch(batch_size)

        data_labes_one_hot = dense_to_one_hot(data_labels,249)

        feed_dict[model.labels] =  data_labels
        feed_dict[model.flow] = data_features
        feed_dict[model.keep_prob] = keep_prob
        feed_dict[model.labels_one_hot] = data_labes_one_hot
        feed_dict[model.learn_rate] = real_lr
        

        
        lr,correct_top_1, correct_top_5, loss, _ = session.run(fetches, feed_dict)

        assert not np.isnan(loss),'loss = NaN'
        total_loss = total_loss + loss

        
        all_correct_top_1 = correct_top_1 + all_correct_top_1
        all_correct_top_5 = correct_top_5 + all_correct_top_5

        if istraining:

            short_term_loss.insert(0,loss)
            if(len(short_term_loss)>short_term_len):
                short_term_loss.pop()
            cur_short_term_loss = sum(short_term_loss) * 1.0 / len(short_term_loss)

            if iter % 5 == 1:
                info = "%.3f: accury(top 1): %.3f accury(top 5): %.3f loss %.3f speed: %.3f sec lr %.5f" % (iter * 1.0 / iterations, correct_top_1 * 1.0 / batch_size, correct_top_5 * 1.0 / batch_size, cur_short_term_loss,
                   (time.time() - start_time) * 1.0 / ((iter + 1) * batch_size),lr )

                print (info)
                log_file.write(info + '\n')
                log_file.flush()

        
        else:
            if iter % 50 == 1:
                info = "%.3f: accury(top 1): %.3f accury(top 5): %.3f loss %.3lf speed: %.3f sec lr %.5f" % (iter * 1.0 / iterations, correct_top_1 * 1.0 / batch_size, correct_top_5 * 1.0 / batch_size, total_loss * 1.0 /(iter + 1),
                   (time.time() - start_time) * 1.0 / ((iter + 1) * batch_size),lr )
                print (info)
                log_file.write(info + '\n')
                log_file.flush()
        
    if istraining:
        return all_correct_top_1 * 1.0 / (iterations * batch_size) , all_correct_top_5 * 1.0 / (iterations * batch_size), total_loss * 1.0 / iterations
    else:
        return all_correct_top_1 * 1.0 / data_size, all_correct_top_5 * 1.0 / data_size, total_loss * 1.0 / iterations

def run_train():
    fout = open('inf.txt','w+')
    train_config = ModelConfig()

    eval_config = ModelConfig()
    eval_config.keep_prob = 1.0
    #eval_config.batch_size = 1

    test_config = ModelConfig()
    test_config.keep_prob = 1.0
    #test_config.batch_size = 1
  
    Session_config = tf.ConfigProto(allow_soft_placement = True)
    Session_config.gpu_options.allow_growth=True 

    
    
    with tf.Graph().as_default(), tf.Session(config=Session_config) as sess:    
        with tf.device('/gpu:1'):
        #if True:
            initializer = tf.random_uniform_initializer(-train_config.init_scale, 
                                                        train_config.init_scale)
            
            train_model = vgg16.Vgg16(FLAGS.vgg16_file_path)
            train_model.build(initializer)

            data_train = dataset.DataSet(FLAGS.file_path_train,FLAGS.data_root_dir,TRAIN_SIZE)
            data_valid = dataset.DataSet(FLAGS.file_path_valid,FLAGS.data_root_dir,VALID_SIZE,is_train_set=False)
            data_test = dataset.DataSet(FLAGS.file_path_test,FLAGS.data_root_dir,TEST_SIZE,is_train_set=False)
            
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    
            saver = tf.train.Saver(max_to_keep=100)
            last_epoch = load_model(sess, saver,FLAGS.saveModelPath,train_model)
            print ('start: ',last_epoch + 1)
    
            for i in range(last_epoch + 1,train_config.max_epoch):

                real_lr = base_lr
                for j in range(i):
                    if j + 1 in down_step:
                        real_lr = real_lr / 2.0
                         
                train_accury_1,train_accury_5,train_loss = run_epoch(real_lr,sess,train_config.keep_prob,fout, train_config.batch_size,train_model, data_train, train_model.train_op,0,train_writer) 
                info = "Epoch: %d Train accury(top 1): %.3f Train accury(top 5): %.3f Loss %.3f" % (i, train_accury_1,train_accury_5,train_loss)
                print (info)
                fout.write(info + '\n')
                fout.flush()
                
                saver.save(sess, FLAGS.saveModelPath + '/model.ckpt', global_step=i)

                valid_accury_1,valid_accury_5,valid_loss = run_epoch(real_lr,sess,eval_config.keep_prob,fout,eval_config.batch_size,train_model, data_valid, tf.no_op(),1,valid_writer,istraining=False) 
                info = "Epoch: %d Valid accury(top 1): %.3f Valid accury(top 5): %.3f Loss: %.3f" % (i, valid_accury_1,valid_accury_5,valid_loss)
                print (info)
                fout.write(info + '\n')
                fout.flush()

            #test_accury_1,test_accury_5,test_loss = run_epoch(real_lr,sess,test_config.keep_prob, fout,test_config.batch_size, train_model, data_test, tf.no_op(),2,test_writer,istraining=False) 
            #info = "Final: Test accury(top 1): %.3f Test accury(top 5): %.3f Loss %.3f" % (test_accury_1,test_accury_5,test_loss)
            #print (info)
            #fout.write(info + '\n')
            #fout.flush()
            
            
            train_writer.close()
            valid_writer.close()
            test_writer.close()

            print("Training step is compeleted!") 
            fout.close() 
            
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
      '--vgg16_file_path',
      type=str,
      default='./pretrain_vgg_16.npy',
      help='file path of vgg16 pretrain model.'
  )
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
      default='/home/share2/chaLearn-Iso/flow/',
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



