from __future__ import print_function
import tensorflow as tf
import glob
import numpy as np
import random
import os

def get_file_list(data_dir, pattern, batch_size):
    # assert os.path.exists(data_dir), 'Directory {} not found.'.format(data_dir)

    file_list = []
    file_glob = os.path.join(data_dir, pattern)
    images = glob.glob(file_glob)
    assert len(images)>=batch_size, 'The batch_size is more than the len of file in {}.'.format(file_glob)

    if len(images)%batch_size != 0:
        file_list.extend(images[:(len(images)//batch_size)*batch_size] + images[-4:])
    else:
        file_list.extend(glob.glob(file_glob))


    file_list.sort()

    return file_list

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    file.truncate()
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()

def tf_read_image(file_name, channels=3, img_type=tf.float32, div_val=255.0, outputsize=None, random_flip_key=None):
    image_string = tf.read_file(file_name)
    image_decoded = tf.div(tf.cast(tf.image.decode_png(image_string, channels=channels), img_type), div_val)
    if outputsize:
        image_decoded = tf.cast(tf.image.resize_images(image_decoded, outputsize, align_corners=True, method=0), img_type)
    if random_flip_key is not None:
        image_decoded = tf.cond(tf.less(random_flip_key[0], .5), lambda: tf.image.flip_up_down(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(random_flip_key[1], .5), lambda: tf.image.flip_left_right(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(random_flip_key[2], .5), lambda: tf.image.transpose_image(image_decoded), lambda: image_decoded)
    return image_decoded



def read_file_image(path, pattern, batch_size):

    imgs = [[],[],[],[]]
    img_names = glob.glob(os.path.join(path, pattern))
    img_names.sort()
    assert (len(img_names)>= batch_size), 'The num of images is not enough in {}.'.format(path)
    for i in range(len(img_names)//batch_size):
        for j in range(len(imgs)):
            imgs[j] += [img_names[i*batch_size+j]]
    if len(img_names) % batch_size != 0:
        for j in range(len(imgs)):
            imgs[j] += [img_names[-1 * batch_size+j]]
    return imgs

def read_flow_image_window(path, pattern, window_size=3, stride=4):

    imgs = []
    k = window_size//2
    for i in range(window_size+stride-1):
        imgs.append([])
    img_names = glob.glob(os.path.join(path, pattern))
    img_names.sort()
    assert (len(img_names)>= stride), 'The num of images is not enough in {}.'.format(path)
    for i in range((len(img_names)-2*k)//stride):
        for j in range(len(imgs)):
            imgs[j] += [img_names[i*stride+j]]
    if (len(img_names)-2*k) % stride != 0:
        for j in range(len(imgs)):
            imgs[j] += [img_names[-1 * len(imgs)+j]]
    return imgs

def read_dataset_flow_window(data_dir, mode, pattern, msk_replace,print_func,time_step=4, window_size=3, shuffle_seed=None):
    imgs, labels = [], []
    for i in range(window_size + time_step - 1):
        imgs.append([])
    for i in range(time_step):
        labels.append([])
    image_names = glob.glob(os.path.join(data_dir, '*'))
    image_names.sort()
    if mode=='train':
        random.seed(shuffle_seed)
        random.shuffle(image_names)
    print_func('The num of ' + mode + ' files: {}'.format(len(image_names)))
    for i in range(len(image_names)):
        n = read_flow_image_window(image_names[i], pattern, window_size=3, stride=time_step)
        for j in range(len(imgs)):
            imgs[j] += n[j]
    instance_num = len(imgs[0])
    print_func('The num of ' + mode + ' slices:{} '.format(instance_num))
    for entry in msk_replace:
        for j in range(len(labels)):
            labels[j] = [name.replace(entry[0], entry[1], 1) for name in imgs[j+1]]

    flows = []
    for i in range(2*time_step):
        flows.append([])
    for i in range(time_step):
        flows[i] = [name.replace('png', 'forward', 1).replace('.png', '.flo', 1) for name in imgs[i+1]]
        flows[i+time_step] = [name.replace('png', 'backward', 1).replace('.png', '.flo', 1) for name in imgs[i]]

    # dataset = tf.data.Dataset.from_tensor_slices((imgs[0],imgs[1],imgs[2],imgs[3],imgs[4],imgs[5],labels[0],labels[1],labels[2],labels[3],\
    #                                                   flows[0],flows[1],flows[2],flows[3],flows[4],flows[5],flows[6],flows[7]))
    imgs = list(transpose(imgs))
    labels = list(transpose(labels))
    flows = list(transpose(flows))
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels, flows))

    return dataset, instance_num

def flow_output(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            data = np.fromfile(f, np.int16, count=2 * w * h)
            data = np.array(data, dtype=np.float32)
            return np.reshape(data, (h, w, 2))

def read_flow(value1):
    return tf.py_func(flow_output, [value1], [tf.float32])

def get_flow_window(imgs, labels, back_for_flows, time_step=4, window_size=3, outputsize=None, random_flip=False):
    random_flip_key = tf.random_uniform([3, ], 0, 1.0) if random_flip else None
    flows, image_decodeds, label_decoded = [], [], []
    flow_num = 2*time_step
    for i in range(flow_num):
        flows.append(read_flow(back_for_flows[i]))
    for i in range(window_size + time_step - 1):
        image_decodeds.append(tf_read_image(imgs[i], outputsize=outputsize))
    for i in range(time_step):
        label_decoded.append(tf_read_image(labels[i], channels=1, img_type=tf.int32, div_val=255, outputsize=outputsize, random_flip_key=random_flip_key))

    names = imgs[(window_size // 2):((window_size // 2) + time_step)]

    return image_decodeds, label_decoded, names, flows

def transpose(matrix):
    return zip(*matrix)

