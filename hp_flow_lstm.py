import os
import sys
import time
import warnings
import numpy as np
import tensorflow as tf
import utils

from utils.models import model_hp_flow_bilstm_stream
from sklearn import metrics
from shutil import rmtree
from operator import itemgetter
from skimage import io
slim = tf.contrib.slim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

img_shape = [240,432]
# dataset
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir', '', 'path to images')
tf.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
tf.flags.DEFINE_string('img_size', '432x240', 'size of input image')
tf.flags.DEFINE_bool('img_aug', False, 'apply image augmentation')
# running configuration
tf.flags.DEFINE_string('mode','train', 'Mode: train / test / visual')
tf.flags.DEFINE_integer('epoch', 8, 'No. of epoch to run')
tf.flags.DEFINE_float('train_ratio', 0.9, 'Trainning ratio')
tf.flags.DEFINE_string('restore', None, 'Explicitly restore checkpoint')
tf.flags.DEFINE_bool('reset_global_step',False, 'Reset global step')
# learning configuration
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_string('optimizer', 'Adam', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_float('lr_decay', 0.5, 'Deca y of learning rate')
tf.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.flags.DEFINE_string('filter_type', 'd1', 'Filter kernel type')
tf.flags.DEFINE_bool('filter_learnable', True, 'Learnable filter kernel')
tf.flags.DEFINE_string('loss', 'dice', 'Loss function type')
tf.flags.DEFINE_float('focal_gamma', 2.0, 'gamma of focal loss')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_integer('shuffle_seed', None, 'Seed for shuffling images')
# logs
tf.flags.DEFINE_string('logdir', '/home/weishujin/Codes/tensorflow/Deep_inpainting_localization/results/h264/CQP23/model_hp_flow_stream2/STTN_FFIT', 'path to logs directory')
tf.flags.DEFINE_integer('verbose_time', 10, 'verbose times in each epoch')
tf.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.flags.DEFINE_integer('keep_ckpt', 1, 'num of checkpoint files to keep')
# outputs
tf.flags.DEFINE_string('visout_dir', None, 'path to output directory')

OPTIMIZERS = {
    'GradientDescent': {'func': tf.train.GradientDescentOptimizer, 'args': {}},
    'Adadelta': {'func': tf.train.AdadeltaOptimizer, 'args': {}},
    'Momentum': {'func': tf.train.MomentumOptimizer, 'args': {'momentum': 0.9}},
    'Adam': {'func': tf.train.AdamOptimizer, 'args': {}},
    'Ftrl': {'func': tf.train.FtrlOptimizer, 'args': {}},
    'RMSProp': {'func': tf.train.RMSPropOptimizer, 'args': {}}
    }
LOSS = {
    'wxent': {'func': utils.losses.sparse_weighted_softmax_cross_entropy_with_logits, 'args': {}},
    'focal':  {'func': utils.losses.focal_loss, 'args': {'gamma': FLAGS.focal_gamma}},
    'xent':  {'func': utils.losses.sparse_softmax_cross_entropy_with_logits, 'args': {}},
    'iou': {'func':utils.losses.iou_coe},
    'dice': {'func':utils.losses.dice_coe},
    'fusion': {'func':utils.losses.loss_fusion}
    }

def main(argv=None):

    if FLAGS.logdir is None:
        sys.stderr.write('Log dir not specified.\n')
        return None
    if FLAGS.mode == 'train':
        write_log_mode = 'w'
        if not os.path.isdir(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        else:
            if os.listdir(FLAGS.logdir):
                sys.stderr.write('Log dir is not empty, continue? [yes(y)/remove(r)/no(n)]: ')
                chioce = input('')
                if (chioce == 'y' or chioce == 'Y'):
                    write_log_mode = 'a'
                elif (chioce == 'r' or chioce == 'R'):
                    rmtree(FLAGS.logdir)
                else:
                    sys.stderr.write('Abort.\n')
                    return None
        tee_print = utils.tee_print.TeePrint(filename=FLAGS.logdir+'.log', mode=write_log_mode)
        print_func = tee_print.write
    else:
        print_func = print
    
    print_func(sys.argv[0])
    print_func('--------------FLAGS--------------')
    for name, val in sorted(FLAGS.flag_values_dict().items(), key=itemgetter(0)):
        if not ('help' in name or name == 'h'):
            print_func('{}: {}'.format(name,val))
    print_func('---------------------------------')

    # Setting up dataset
    shuffle_seed = FLAGS.shuffle_seed or int(time.time()*256)
    print_func('Seed={}'.format(shuffle_seed))
    if 'jpg' in FLAGS.data_dir:
        pattern = '*.jpg'
        msk_rep = [['jpg','msk'],['.jpg','.png']]
    else:
        pattern = '*.png'
        msk_rep = [['png','msk']]

    def map_func(imgs,label,for_back_flows):
            return utils.read_dataset.get_flow_window(imgs,label,for_back_flows, window_size=3, outputsize=[int(v) for v in reversed( FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

    if FLAGS.mode == 'train':
        dataset, instance_num = utils.read_dataset.read_dataset_flow_window(FLAGS.data_dir, mode=FLAGS.mode,pattern=pattern,\
                                                                            msk_replace=msk_rep, shuffle_seed=shuffle_seed, print_func=print_func)
        dataset_trn = dataset.shuffle(buffer_size=100000).map(map_func).batch(FLAGS.batch_size, drop_remainder=True).repeat()
        FLAGS.data_dir = (FLAGS.data_dir).replace('train','val',1)
        dataset, _ = utils.read_dataset.read_dataset_flow_window(FLAGS.data_dir, mode='val', pattern=pattern,\
                                                                 msk_replace=msk_rep, shuffle_seed=shuffle_seed,  print_func=print_func)
        dataset_vld = dataset.map(map_func).batch(FLAGS.batch_size, drop_remainder=True)
        iterator_trn = dataset_trn.make_one_shot_iterator()
        iterator_vld = dataset_vld.make_initializable_iterator()
    elif FLAGS.mode == 'test' or FLAGS.mode == 'visual':
        dataset, instance_num = utils.read_dataset.read_dataset_flow_window(FLAGS.data_dir, mode=FLAGS.mode,pattern=pattern,\
                                                                            msk_replace=msk_rep, shuffle_seed=shuffle_seed, print_func=print_func)
        dataset_vld = dataset.map(map_func).batch(FLAGS.batch_size, drop_remainder=True)
        iterator_vld = dataset_vld.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, iterator_vld.output_types, iterator_vld.output_shapes)

    next_element = iterator.get_next()
    images = next_element[0]
    labels = next_element[1]
    labels = tf.reshape(labels,[FLAGS.batch_size*4, img_shape[0], img_shape[1]])
    imgnames = next_element[2]
    imgnames = tf.reshape(imgnames,[-1,])
    flows = next_element[3]

    is_training = tf.placeholder(tf.bool, [])
    logits, preds, preds_map = model_hp_flow_bilstm_stream(images, flows, FLAGS.batch_size, FLAGS.filter_type, FLAGS.filter_learnable, FLAGS.weight_decay, is_training, img_shape)
    loss = LOSS[FLAGS.loss]['func'](logits=logits, labels=labels) + tf.add_n(tf.losses.get_regularization_losses())
    global_step = tf.Variable(0, trainable=False, name='global_step')
    itr_per_epoch = int(np.ceil(instance_num) // FLAGS.batch_size)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps=int(itr_per_epoch*FLAGS.lr_decay_freq),decay_rate=FLAGS.lr_decay,staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = OPTIMIZERS[FLAGS.optimizer]['func'](learning_rate,**OPTIMIZERS[FLAGS.optimizer]['args']).\
                    minimize(loss, global_step=global_step, var_list=tf.trainable_variables())
    
    with tf.name_scope('metrics'):
        tp_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,1),tf.equal(preds,1))),name='true_positives')
        tn_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,0),tf.equal(preds,0))),name='true_negatives')
        fp_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,0),tf.equal(preds,1))),name='false_positives')
        fn_count  = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels,1),tf.equal(preds,0))),name='false_negatives')
        metrics_count = tf.Variable(0.0, name='metrics_count', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        Groundtruth_zeros_count = tf.Variable(0.0, name='Groundtruth_zeros_count', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        recall_sum    = tf.Variable(0.0, name='recall_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        precision_sum = tf.Variable(0.0, name='precision_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        accuracy_sum  = tf.Variable(0.0, name='accuracy_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        loss_sum      = tf.Variable(0.0, name='loss_sum', trainable = False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        update_zeros_count = tf.assign_add(Groundtruth_zeros_count, tf.cond(tf.equal(tp_count + fn_count, 0), \
                                                                            lambda: 1.0, \
                                                                            lambda: 0.0))
        # update_recall_sum = tf.assign_add(recall_sum, tp_count/(tp_count+fn_count))
        update_recall_sum = tf.assign_add(recall_sum, tf.cond(tf.equal(tp_count + fn_count, 0), \
                                                              lambda: 0.0, \
                                                              lambda: tp_count / (tp_count + fn_count)))
        update_precision_sum = tf.assign_add(precision_sum, tf.cond(tf.equal(tp_count+fp_count,0), \
                                                                    lambda: 0.0, \
                                                                    lambda: tp_count/(tp_count+fp_count)))
        update_accuracy_sum = tf.assign_add(accuracy_sum, (tp_count+tn_count)/(tp_count+tn_count+fp_count+fn_count))
        update_loss_sum = tf.assign_add(loss_sum, loss)
        with tf.control_dependencies([update_zeros_count, update_recall_sum, update_precision_sum, update_accuracy_sum, update_loss_sum]):
            update_metrics_count = tf.assign_add(metrics_count, 1.0)
        mean_recall = recall_sum / (metrics_count - Groundtruth_zeros_count)
        mean_precision = precision_sum / (metrics_count - Groundtruth_zeros_count)
        mean_accuracy = accuracy_sum/metrics_count
        mean_loss = loss_sum/metrics_count
    
    config=tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    local_vars_metrics = [v for v in tf.local_variables() if 'metrics/' in v.name]

    saver = tf.train.Saver(max_to_keep= FLAGS.keep_ckpt+1 if FLAGS.keep_ckpt else 1000000)
    model_checkpoint_path = ''
    if FLAGS.restore and 'ckpt' in FLAGS.restore:
        model_checkpoint_path = FLAGS.restore
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.restore or FLAGS.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path

    if model_checkpoint_path:
        saver.restore(sess, model_checkpoint_path)
        print_func('Model restored from {}'.format(model_checkpoint_path))

    if FLAGS.mode == 'train':
        summary_op = tf.summary.merge([tf.summary.scalar('loss', mean_loss),
                                       tf.summary.scalar('lr', learning_rate)])
        summary_writer_trn = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
        summary_writer_vld = tf.summary.FileWriter(FLAGS.logdir + '/validation')

        handle_trn = sess.run(iterator_trn.string_handle())
        handle_vld = sess.run(iterator_vld.string_handle())


        best_metric = 0.0
        if FLAGS.reset_global_step:
            sess.run(tf.variables_initializer([global_step]))
        for itr in range(itr_per_epoch*FLAGS.epoch):
            _, step, _, = sess.run([train_op, global_step, update_metrics_count], feed_dict={handle: handle_trn, is_training: True})
            if step % (itr_per_epoch//FLAGS.verbose_time) == 0:
                mean_loss_, mean_accuracy_, mean_recall_, mean_precision_, summary_str = sess.run([mean_loss, mean_accuracy, mean_recall, mean_precision, summary_op])
                print_func('epoch: {:d} step: {:d} loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Precision: {:0.6f}'.format(\
                            int(step//itr_per_epoch),step,mean_loss_,mean_accuracy_,mean_recall_,mean_precision_))
                summary_writer_trn.add_summary(summary_str, step)
                sess.run(tf.variables_initializer(local_vars_metrics))
            if step > 0 and step % (itr_per_epoch//FLAGS.valid_time) == 0:
                sess.run(iterator_vld.initializer)
                sess.run(tf.variables_initializer(local_vars_metrics))
                TNR, F1, MCC, IoU, Recall, Prec, Auc = [], [], [], [], [], [], []
                warnings.simplefilter('ignore',RuntimeWarning)
                while True:
                    try:
                        labels_, preds_, _ = sess.run([labels, preds, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
                        for i in range(labels_.shape[0]):
                            recall, tnr, prec, f1, mcc, iou, fn, tp,_ = utils.metrics.get_metrics(labels_[i],preds_[i])
                            auc = metrics.roc_auc_score(labels_[i].reshape(-1), preds_[i].reshape(-1))
                            TNR.append(tnr)
                            F1.append(f1)
                            MCC.append(mcc)
                            IoU.append(iou)
                            Recall.append(recall)
                            Prec.append(prec)
                            Auc.append(auc)
                    except tf.errors.OutOfRangeError:
                        break
                mean_loss_, mean_accuracy_, summary_str = sess.run([mean_loss, mean_accuracy, summary_op])
                if np.mean(F1) > best_metric:
                    best_metric = np.mean(F1)
                print_func('validation loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f} best_metric: {:0.6f}'.format( \
                            mean_loss_,mean_accuracy_,np.mean(Recall),np.mean(Prec),np.mean(TNR),np.mean(F1),np.mean(MCC),np.mean(IoU),best_metric))
                summary_writer_vld.add_summary(summary_str, step)
                sess.run(tf.variables_initializer(local_vars_metrics))

                saver.save(sess, '{}/model.ckpt-{:0.6f}'.format(FLAGS.logdir, np.mean(F1)), int(step/itr_per_epoch))
                saver._last_checkpoints = sorted(saver._last_checkpoints, key=lambda x: x[0].split('-')[1])
                if FLAGS.keep_ckpt and len(saver._last_checkpoints) > FLAGS.keep_ckpt:
                    saver._checkpoints_to_be_deleted.append(saver._last_checkpoints.pop(0))
                    saver._MaybeDeleteOldCheckpoints()
                tf.train.update_checkpoint_state(save_dir=FLAGS.logdir, \
                    model_checkpoint_path=saver.last_checkpoints[-1], \
                    all_model_checkpoint_paths=saver.last_checkpoints)

    elif FLAGS.mode == 'test':
        def test(preds_map_, labels_, threshold, TNR, F1, MCC, IoU, Recall, Prec, Fpr):
            pred_ = np.array(preds_map_)
            pred_[pred_ < threshold] = 0
            pred_[pred_ >= threshold] = 1
            recall, tnr, prec, f1, mcc, iou, fn, tp, fpr = utils.metrics.get_metrics(labels_, pred_)
            TNR.append(tnr)
            F1.append(f1)
            MCC.append(mcc)
            IoU.append(iou)
            Recall.append(recall)
            Prec.append(prec)
            Fpr.append(fpr)

        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        TNR, F1, MCC, IoU, Recall, Prec, Fpr = [], [], [], [], [], [], []
        for i in range(11):
            TNR.append([]), F1.append([]), MCC.append([]), IoU.append([]), Recall.append([]), Prec.append([]), Fpr.append([])
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        while True:
            try:
                labels_, preds_, preds_map_, imgnames_, _ = sess.run([labels, preds, preds_map, imgnames, update_metrics_count], feed_dict={handle: handle_vld, is_training: False})
                for i in range(labels_.shape[0]):
                    for j in range(len(F1)):
                        test(preds_map_[i], labels_[i], 0.1*j, TNR[j], F1[j], MCC[j], IoU[j], Recall[j], Prec[j], Fpr[j])
            except tf.errors.OutOfRangeError:
                break
        mean_loss_, mean_accuracy_ = sess.run([mean_loss, mean_accuracy])
        for j in range(len(F1)):
            print_func('{} testing loss: {:0.6f} ACC: {:0.6f} Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f} Fpr: {:0.6f}'.format( \
                    j,mean_loss_, mean_accuracy_, np.mean(Recall[j]), np.mean(Prec[j]), np.mean(TNR[j]), np.mean(F1[j]),np.mean(MCC[j]), np.mean(IoU[j]), np.mean(Fpr[j])))

    elif FLAGS.mode == 'visual':
        handle_vld = sess.run(iterator_vld.string_handle())
        sess.run(iterator_vld.initializer)
        warnings.simplefilter('ignore',(UserWarning, RuntimeWarning))
        if not os.path.exists(FLAGS.visout_dir):
            os.makedirs(FLAGS.visout_dir)
        index = 0
        img_files = ''
        img_f1, img_fpr = 0, 0
        img_f1_index, img_fpr_index=0,0
        while True:
            try:
                labels_, preds_, preds_map_, imgnames_, images_ = sess.run([labels, preds, preds_map, imgnames, visual_images], feed_dict={handle: handle_vld, is_training: False})
                for i in range(labels_.shape[0]):
                    imgname = imgnames_[i].decode().split('/')[-1]
                    vis_out = preds_map_[i]
                    vis_out = vis_out[...,np.newaxis]
                    pre_out = np.array(vis_out)
                    pre_out[pre_out < 0.5] = 0
                    pre_out[pre_out >= 0.5] = 1
                    lab_out = labels_[i]
                    lab_out = lab_out[...,np.newaxis]
                    Mg, Mo, Mo_ = lab_out * 255, vis_out * 255, pre_out * 255
                    H = img_shape[0]
                    W = img_shape[1]
                    out = np.zeros([H * 4, W, 3])
                    out[:H, :, :] = images_[i] * 255
                    out[H:H * 2, :, :] = np.concatenate([Mo, Mo, Mo], axis=2)
                    out[H * 2:H * 3, :, :] = np.concatenate([Mo_, Mo_, Mo_], axis=2)
                    out[H * 3:, :, :] = np.concatenate([Mg, Mg, Mg], axis=2)

                    if not os.path.exists(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'inpainting')):
                        os.makedirs(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'inpainting'))
                    if not os.path.exists(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'pred')):
                        os.makedirs(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'pred'))
                    if not os.path.exists(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'pred_mask')):
                        os.makedirs(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'pred_mask'))
                    if not os.path.exists(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'groundth')):
                        os.makedirs(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2],'groundth'))
                    io.imsave(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2], 'inpainting',imgname.replace('.png','_inpainting.png')), np.uint8(out[:H, :, :]))
                    io.imsave(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2], 'pred',imgname.replace('.png', '_pred.png')), np.uint8(out[H:2*H, :, :]))
                    io.imsave(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2], 'pred_mask',imgname.replace('.png', '_pred_mask.png')), np.uint8(out[2*H:3*H, :, :]))
                    io.imsave(os.path.join(FLAGS.visout_dir, imgnames_[i].decode().split('/')[-2], 'groundth',imgname.replace('.png', '_groundth.png')), np.uint8(out[3*H:, :, :]))
                    # cv2.imwrite(os.path.join(FLAGS.visout_dir,imgname.replace('.png','_pred.png')), out)
                    # vis_out = labels_[i]
                    # io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_gt.png')), np.uint8(np.round(vis_out*255.0)))
                    # vis_out = images_[i]
                    # io.imsave(os.path.join(FLAGS.visout_dir,imgname.replace('.jpg','_img.png')), np.uint8(np.round(vis_out*255.0)))
                    recall,tnr,prec,f1,mcc,iou,fn,tp,fpr = utils.metrics.get_metrics(labels_[i],preds_[i])
                    # print('{}: {} '.format(index,imgnames_[i].decode().split('/')[-2] + '_' + imgname),end='')
                    if img_files != imgnames_[i].decode().split('/')[-2]:
                        if img_files!='':
                            print('{} {:0.3f}'.format(img_files, img_fpr / img_fpr_index))
                        img_files = imgnames_[i].decode().split('/')[-2]
                        img_f1, img_fpr = 0, 0
                        img_f1_index, img_fpr_index = 0, 0
                    # if fn + tp != 0:
                    #     # print('Recall: {:0.6f} Prec: {:0.6f} TNR: {:0.6f} \033[1;31mF1: {:0.6f}\033[0m MCC: {:0.6f} IoU: {:0.6f}'.format(recall,prec,tnr,f1,mcc,iou),end='')
                    #     img_f1_index += 1
                    #     img_f1 += f1
                    img_fpr += fpr
                    img_fpr_index += 1
                    # else:
                    #     # print('Groundtruth is all zeros', end='')

                    index += 1
                    # print('')
            except tf.errors.OutOfRangeError:
                print('{} {:0.3f}'.format(img_files, img_fpr / img_fpr_index))
                break

    else:
        print_func('Mode not defined: '+FLAGS.mode)
        return None

if __name__ == '__main__':
    tf.app.run()
