import tensorflow._api.v2.compat.v1 as tf#加上这两句才能在tensorflow2.x运行1.x
tf.disable_v2_behavior()

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
