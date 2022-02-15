import numpy as np
import tensorflow as tf

tf_ver = tf.__version__.split('.')
if int(tf_ver[0])<=1 and int(tf_ver[1])<=4:
    softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits
else:
    softmax_cross_entropy_with_logits = lambda labels=None, logits=None: tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits)

def sparse_weighted_softmax_cross_entropy_with_logits(logits, labels, num_classes=2):
    logits = tf.reshape(logits, (-1, num_classes))

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot_labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    # compute weights
    occurrence = tf.cast(tf.bincount(label_flat, minlength=num_classes, maxlength=num_classes), tf.float32)
    class_weight =  tf.cond(tf.equal(tf.count_nonzero(occurrence), num_classes), \
                            lambda: tf.expand_dims(tf.div(tf.reduce_mean(occurrence), occurrence), axis=1), \
                            lambda: tf.ones(shape=[num_classes,1]))
    weights = tf.squeeze(tf.matmul(one_hot_labels, class_weight))

    loss = tf.multiply(weights, softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
    loss = tf.reduce_mean(loss)
    return loss                                                                         

def focal_loss(logits, labels, gamma=2.0, num_classes=2):
    logits = tf.reshape(logits, (-1, num_classes))

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot_labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    
    # compute weights
    weights = tf.reduce_sum(tf.multiply(one_hot_labels,tf.pow(tf.subtract(1.0, tf.nn.softmax(logits)), gamma)),1)

    loss = tf.multiply(weights, softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
    loss = tf.reduce_mean(loss)
    return loss

def sparse_softmax_cross_entropy_with_logits(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))

# def dice_coe(logits, labels, num_classes=2,cost_name="dice_coe", axis=[1, 2, 3], threshold=0.5, smooth=1e-5):
#     """
#     s = 2|A∩B|/(|A|+|B|)
#     d = 1 - s
#     """
#     y_pred = tf.cast(logits, tf.float32)
#     y_true = tf.one_hot(labels, depth=num_classes)
#     if cost_name == "dice_coe":
#         loss_type = "jaccard"
#         intersection = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection A∩B
#         if loss_type == "jaccard":  # default loss type, in fact, jaccard and soresen are the same thing
#             A = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in y_true
#             B = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in y_pred
#         elif loss_type == "sorensen":
#             A = tf.reduce_sum(y_true, axis=axis)
#             B = tf.reduce_sum(y_pred, axis=axis)
#         else:
#             raise Exception("Unknow loss_type")
#         dice = (2.0 * intersection) / (A + B + smooth)  # compute dice coefficient
#         loss = 1 - tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
#
#     return loss

# def iou_coe(logits, labels, num_classes=2,threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
#     """Non-differentiable Intersection over Union (IoU) for comparing the
#     similarity of two batch of data, usually be used for evaluating binary image segmentation.
#     The coefficient between 0 to 1, and 1 means totally match.
#
#     Parameters
#     -----------
#     logits : tensor
#         A batch of distribution with shape: [batch_size, ....], (any dimensions).
#     labels : tensor
#         The labels distribution, format the same with `logits`.
#     threshold : float
#         The threshold value to be true.
#     axis : tuple of integer
#         All dimensions are reduced, default ``(1,2,3)``.
#     smooth : float
#         This small value will be added to the numerator and denominator, see ``dice_coe``.
#
#     Notes
#     ------
#     - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.
#
#     """
#     labels = tf.one_hot(labels, depth=num_classes)
#     pre = tf.cast(logits > threshold, dtype=tf.float32)
#     truth = tf.cast(labels > threshold, dtype=tf.float32)
#     inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
#     union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
#     ## old axis=[0,1,2,3]
#     # epsilon = 1e-5
#     # batch_iou = inse / (union + epsilon)
#     ## new haodong
#     batch_iou = (inse + smooth) / (union + smooth)
#     iou = tf.reduce_mean(batch_iou)
#     return iou  #, pre, truth, inse, union

def dice_coe(logits, labels, loss_type='jaccard', smooth=1e-5,num_classes = 2):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    # >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    # >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    logits = tf.nn.softmax(tf.reshape(logits,(-1,num_classes)))
    # labels = tf.cast(labels,dtype = tf.float32)
    labels_flat = tf.reshape(labels,(-1,1))
    one_hot_labels = tf.reshape(tf.one_hot(labels_flat, depth=num_classes), (-1, num_classes))

    logits = logits[:,1]
    labels = one_hot_labels[:,1]
    inse = tf.reduce_sum(logits * labels)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(logits * logits)
        r = tf.reduce_sum(labels * labels)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(logits)
        r = tf.reduce_sum(labels)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = 1 - tf.reduce_mean(dice)
    return dice


def loss_fusion(logits, labels):
    return focal_loss(logits, labels) + iou_coe(logits, labels) + ssim(logits, labels)

def ssim(logits, labels):
    logits = tf.nn.softmax(logits)
    img1 = logits[..., 1]
    # img1 = tf.gather(logits, axis=-1, indices=[1])
    img2 = tf.cast(labels, tf.float32)
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    loss = 1 - tf.reduce_mean(tf.image.ssim(img2, img1, max_val=1.0))

    return loss

def iou_coe(logits, labels, num_classes=2,threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    logits = tf.nn.softmax(logits)
    labels = tf.one_hot(labels, depth=num_classes)
    pre = tf.cast(logits > threshold, dtype=tf.float32)
    truth = tf.cast(labels > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou  #, pre, truth, inse, union

def focal_loss(logits, labels, gamma=2.0, num_classes=2):
    logits = tf.reshape(logits, (-1, num_classes))

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot_labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    # compute weights
    weights = tf.reduce_sum(tf.multiply(one_hot_labels, tf.pow(tf.subtract(1.0, tf.nn.softmax(logits)), gamma)), 1)

    loss = tf.multiply(weights, softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
    loss = tf.reduce_mean(loss)
    return loss



