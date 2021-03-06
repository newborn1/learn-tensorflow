#gpu和cpu版本是不能同时装的,并且如果不做处理的话2.x版本不兼容1.x
import tensorflow as tf
tf.compat.v1.disable_eager_execution()#2不兼容1，这里才能用sess.run().ERROR:The Session graph is empty.  Add operations to the graph before calling run().

with tf.device('/cpu:0'):
    a = tf.constant ([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant ([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:0'):#gpu:0?gpu:1?
    c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto (allow_soft_placement=True, log_device_placement=True))
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.compat.v1.global_variables_initializer())
print (sess.run(c))