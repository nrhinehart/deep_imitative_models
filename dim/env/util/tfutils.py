import dill
import tensorflow as tf

def save_feed_dict(fd, filename):
    names = {k.name: v for k, v in fd.items()}
    with open(filename, 'wb') as f:
        dill.dump(names, f)
    return names

def safe_divide(numerator, denominator):
    x = denominator
    x_ok = tf.not_equal(x, 0.)
    f = lambda x: numerator / x
    safe_f = tf.zeros_like
    safe_x = tf.where(x_ok, x, tf.ones_like(x))
    u_safe = tf.where(x_ok, f(safe_x), safe_f(x))
    return u_safe
