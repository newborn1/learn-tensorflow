# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf#加上这两句才能在tensorflow2.x运行1.x
tf.disable_v2_behavior()
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

"""搭建神经网络的结构"""
#搭建模型
Weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

#计算误差
loss = tf.reduce_mean(tf.square(y-y_data))#square()是计算平方

#传播误差
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

"""把数据应用于搭建好的网络结构中"""
#初始化之前定义的Variable变量
init = tf.global_variables_initializer()

#创建会话Session
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))