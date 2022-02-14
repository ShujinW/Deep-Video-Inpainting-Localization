import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import utils.resnet_v2_IN as resnet_v2
from utils.bilinear_upsample_weights import bilinear_upsample_weights
from utils.cell import ConvLSTMCell

FILTERS = {
    'd1': [
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]),
        np.array([[0., 0., 0.], [0., -1., 0.], [0., 0., 1.]])],
    'd2': [
        np.array([[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]]),
        np.array([[1., 0., 0.], [0., -2., 0.], [0., 0., 1.]])],
    'd3': [
        np.array([[0., 0., 0., 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., -3., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -1., 3., -3., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., -1., 0., 0., 0.], [0., 0., 3., 0., 0.], [0., 0., 0., -3., 0.], [0., 0., 0., 0., 1.]])],
    'd4': [
        np.array([[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
        np.array([[1., 0., 0., 0., 0.], [0., -4., 0., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., 0., -4., 0.], [0., 0., 0., 0., 1.]])]
    }
def get_residuals(image, filter_type='d1', filter_trainable=True, image_channel=3):
    if filter_type == 'none':
        return image - np.array([123.68, 116.78, 103.94]) / 255.0

    residuals = []

    if filter_type == 'random':
        for kernel_index in range(3):
            kernel_variable = tf.get_variable(name='root_filter{}'.format(kernel_index), shape=[3, 3, image_channel, 1], \
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            image_filtered = tf.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
    else:
        kernel_index = 0
        for filter_kernel in FILTERS[filter_type]:
            kernel_variable = tf.Variable(np.repeat(filter_kernel[:, :, np.newaxis, np.newaxis], image_channel, axis=2), \
                                          trainable=filter_trainable, dtype='float',name='root_filter{}'.format(kernel_index))
            image_filtered = tf.nn.depthwise_conv2d_native(image, kernel_variable, strides=[1, 1, 1, 1], padding='SAME')
            residuals.append(image_filtered)
            kernel_index += 1

    return tf.concat(residuals, 3)

def resnet_block(inputs,
                 blocks,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 reuse=None,
                 scope='resnet_small1'):
    return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                               global_pool=global_pool, output_stride=output_stride,
                               include_root_block=include_root_block,
                               reuse=reuse, scope=scope)


def bi_convlstm(net, batch,image_shape, output_channels, convlstm_name, time_step=4):

    net = tf.reshape(net, [-1, time_step, image_shape[0], image_shape[1], image_shape[2]])
    cell1 = ConvLSTMCell([image_shape[0], image_shape[1]], output_channels, [3, 3])
    cell2 = ConvLSTMCell([image_shape[0], image_shape[1]], output_channels, [3, 3])
    initial_state1 = cell1.zero_state(batch_size=batch, dtype=tf.float32)
    initial_state2 = cell2.zero_state(batch_size=batch, dtype=tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, net, dtype=net.dtype, time_major=False, initial_state_fw=initial_state1, initial_state_bw=initial_state2,scope=convlstm_name)
    output_fw, output_bw = outputs
    lstm_logits = tf.concat([output_fw, output_bw], axis=-1)
    lstm_logits_squeeze = tf.reshape(lstm_logits, [-1, image_shape[0], image_shape[1], output_channels*2])
    return lstm_logits_squeeze

def multi_calculate_residual_window(image, flow, h, w, batch,time_step=4):
    if time_step == 3:
        img1 = tf.gather(image, axis=1, indices=[1, 2, 3])
        img2 = tf.gather(image, axis=1, indices=[2, 3, 4])
        img3 = tf.gather(image, axis=1, indices=[0, 1, 2])
    elif time_step == 4:
        img1 = tf.gather(image, axis=1, indices=[1, 2, 3, 4])
        img2 = tf.gather(image, axis=1, indices=[2, 3, 4, 5])
        img3 = tf.gather(image, axis=1, indices=[0, 1, 2, 3])
    elif time_step == 5:
        img1 = tf.gather(image, axis=1, indices=[1, 2, 3, 4, 5])
        img2 = tf.gather(image, axis=1, indices=[2, 3, 4, 5, 6])
        img3 = tf.gather(image, axis=1, indices=[0, 1, 2, 3, 4])
    img11 = tf.concat([img1, img1], axis=1)
    img23 = tf.concat([img2, img3], axis=1)
    img1 = tf.reshape(img11, [-1, h, w, 3])
    img2 = tf.reshape(img23, [-1, h, w, 3])

    range_x = tf.range(h, dtype=tf.float32)
    range_y = tf.range(w, dtype=tf.float32)
    range_x = tf.reshape(range_x, (-1, 1))
    range_y = tf.reshape(range_y, (1, -1))
    range_x = tf.tile(range_x, [1, w])
    range_y = tf.tile(range_y, [h, 1])

    img1 = tf.reshape(img1, (-1, h * w, 3))
    img2 = tf.reshape(img2, (-1, h * w, 3))

    x = tf.cast(tf.round(flow[..., 1] + range_x), dtype=tf.int32)
    y = tf.cast(tf.round(flow[..., 0] + range_y), dtype=tf.int32)
    x = tf.reshape(x, (-1, h * w))
    y = tf.reshape(y, (-1, h * w))
    c = tf.cast((x >= 0) & (x < h) & (y >= 0) & (y < w), dtype=tf.int32)
    x_y = tf.add(tf.multiply(x, w), y)
    x_y = tf.multiply(x_y, c)

    def gather(img2_xy):
        img2 = img2_xy[0]
        xy = img2_xy[1]
        return tf.gather(img2, xy, axis=0)

    img2_wap = tf.map_fn(gather, (img2, x_y), dtype=tf.float32)

    img_subtract = tf.subtract(img1, img2_wap)
    img_subtract = tf.multiply(img_subtract, tf.cast(tf.reshape(c, [batch * time_step*2, -1, 1]), tf.float32))
    img_subtract = tf.reshape(img_subtract, (batch, time_step*2, h, w, 3))

    if time_step == 3:
        img_subtract = tf.concat([tf.gather(img_subtract, axis=1, indices=[0,1,2]), tf.gather(img_subtract, axis=1, indices=[3,4,5])], axis=-1)
    elif time_step == 4:
        img_subtract = tf.concat([tf.gather(img_subtract, axis=1, indices=[0,1,2,3]), tf.gather(img_subtract, axis=1, indices=[4,5,6,7])],axis=-1)
    elif time_step == 5:
        img_subtract = tf.concat([tf.gather(img_subtract, axis=1, indices=[0,1,2,3,4]),tf.gather(img_subtract, axis=1, indices=[5,6,7,8,9])], axis=-1)
    img_subtract = tf.reshape(img_subtract, (batch*time_step, h, w, 6))
    return img_subtract

def model_hp_flow_bilstm_stream(images, flows, batch_size, filter_type, filter_trainable, weight_decay, is_training, img_shape, time_step=4, num_classes=2):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        images = tf.reshape(images, [batch_size, time_step+2, img_shape[0],img_shape[1], 3])
        if time_step == 3:
            image1 = tf.gather(images, axis=1, indices=[1, 2, 3])
        elif time_step == 4:
            image1 = tf.gather(images, axis=1, indices=[1, 2, 3, 4])
        elif time_step == 5:
            image1 = tf.gather(images, axis=1, indices=[1, 2, 3, 4, 5])
        image1 = tf.reshape(image1, [-1, img_shape[0],img_shape[1], 3])
        inputs1 = get_residuals(image1, filter_type, filter_trainable)
        inputs2 = multi_calculate_residual_window(images, flows, img_shape[0],img_shape[1], batch_size,time_step=time_step)

        blocks1 = [
            resnet_v2.resnet_v2_block('block1', base_depth=16, num_units=2, stride=2),
            resnet_v2.resnet_v2_block('block2', base_depth=32, num_units=2, stride=2)
        ]
        _, inputs1 = resnet_block(inputs=inputs1, blocks=blocks1, num_classes=None, is_training=is_training, global_pool=False,\
                                  output_stride=None, include_root_block=False,scope='resnet_block1')
        blocks2 = [
            resnet_v2.resnet_v2_block('block1', base_depth=16, num_units=2, stride=2),
            resnet_v2.resnet_v2_block('block2', base_depth=32, num_units=2, stride=2)
        ]
        _, inputs2 = resnet_block(inputs=inputs2, blocks=blocks2, num_classes=None, is_training=is_training,global_pool=False, \
                                  output_stride=None, include_root_block=False, scope='resnet_block2')

        net = tf.concat([inputs1['resnet_block1/block2'], inputs2['resnet_block2/block2']], axis=-1)

        blocks3 = [
            resnet_v2.resnet_v2_block('block1', base_depth=128, num_units=2, stride=2),
            resnet_v2.resnet_v2_block('block2', base_depth=256, num_units=2, stride=2)
        ]
        _, net = resnet_block(inputs=net, blocks=blocks3, num_classes=None, is_training=is_training,global_pool=False, \
                                  output_stride=None, include_root_block=False, scope='resnet_block3')

        net = bi_convlstm(net['resnet_block3/block2'], batch_size, [img_shape[0]//16,img_shape[1]//16, 1024], 512, 'convlstm1',time_step)
        net = tf.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,64,1024),dtype=tf.float32,name='bilinear_kernel0'), \
                                     [batch_size*time_step, tf.shape(image1)[1]//4,tf.shape(image1)[2]//4, 64], strides=[1, 4, 4, 1], padding="SAME")
        net = bi_convlstm(net, batch_size, [img_shape[0]//4,img_shape[1]//4, 64], 32, 'convlstm2',time_step)
        net = tf.nn.conv2d_transpose(net, tf.Variable(bilinear_upsample_weights(4,4,64),dtype=tf.float32,name='bilinear_kernel1'), \
                                     [batch_size*time_step, tf.shape(image1)[1], tf.shape(image1)[2], 4], strides=[1, 4, 4, 1], padding="SAME")

        net = tf.contrib.layers.instance_norm(net, activation_fn=tf.nn.relu, scope='post_norm')
        logits = slim.conv2d(net, num_classes, [5, 5], activation_fn=None, normalizer_fn=None, scope='logits')
        preds = tf.cast(tf.argmax(logits,3),tf.int32)
        preds_map = tf.nn.softmax(logits)[:,:,:,1]

        return logits, preds, preds_map
