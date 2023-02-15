from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import segmentation_models as sm


# Keras MeanIoU
# ----------------------------------------------------------------------------------------------
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=3), tf.argmax(y_pred, axis=3), sample_weight)


# Keras categorical accuracy
# ----------------------------------------------------------------------------------------------
def cat_acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true,y_pred)


# Custom dice coefficient metric
# ----------------------------------------------------------------------------------------------
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_score(y_true, y_pred):
    return dice_coef(y_true, y_pred)


# Keras AUC metric
# ----------------------------------------------------------------------------------------------
def auc():
    return tf.keras.metrics.AUC(num_thresholds=3)


# Custom jaccard score
# ----------------------------------------------------------------------------------------------
def jaccard_score(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
    return (jac) * smooth + tf.keras.losses.binary_crossentropy(y_true, y_pred)


# Metrics
# ----------------------------------------------------------------------------------------------

def get_metrics(config):
    """
    Summary:
        create keras MeanIoU object and all custom metrics dictornary
    Arguments:
        config (dict): configuration dictionary
    Return:
        metrics directories
    """

    
    m = MyMeanIOU(config['num_classes'])
    return {
            'my_mean_iou': m,
            'f1_score':sm.metrics.f1_score,
            'precision':sm.metrics.precision,
            'recall':sm.metrics.recall,
            'dice_coef_score':dice_coef_score
            #'cat_acc':cat_acc # reduce mean_iou
          }
#metrics = ['acc']
