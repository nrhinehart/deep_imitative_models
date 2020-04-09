
import pdb
import tensorflow as tf

from tensorflow.python.framework import op_def_registry as registry

def load_old_model(checkpoint_path, sess):
    saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
    pdb.set_trace()
