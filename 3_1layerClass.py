#https://www.tensorflow.org/versions/r2.5/api_docs/python/tf
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras import layers


import numpy as np
import matplotlib.pyplot as plt

#超参数
PROB = 0.7#这里有效果,点太少了，不能设太小
LR = 0.01
IN_SIZE = 1
OUT_SIZE = 1
EPOCH = 200
writer = tf.summary.create_file_writer("./graph/logs2/",name="class net")

#定义网络结构（只有一层）
class LinearLayer(layers.Layer) :
    def __init__(self, units = 32, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        
        self.lr = LR
        self.units = units

    def build(self, input_shape):#添加本层要学习的参数,这样就不用收到设置输入的size
        with tf.name_scope('layer'):
            with tf.name_scope('Weights'):
                self.Weights = self.add_weight(
                    shape=(input_shape[-1],self.units),
                    initializer="random_normal",
                    trainable=True,
                )
                with writer.as_default():
                    tf.summary.histogram('Weights',self.Weights,step=self.step)

            with tf.name_scope('biases'):
                self.biases = self.add_weight(
                    shape=(self.units,),
                    initializer="random_normal",
                    trainable=True,
                )
                with writer.as_default():
                    tf.summary.histogram('biases',self.biases,step=self.step)

        return super().build(input_shape)

    def call(self,inputs,step):#类似forward
        with tf.name_scope('outputs'):
            outputs = tf.matmul(inputs,self.Weights)+self.biases
            outputs = activations.relu(outputs,alpha=0.5)#alpha很重要

        losses = self.lr * tf.reduce_sum(inputs)#改？
        self.add_loss(losses)

        with writer.as_default():
            tf.summary.histogram('output',outputs,step=step) 

        return outputs

    def get_config(self):
        config = super(Net,self).get_config()
        config.update({"units":self.units})

        return config

class Net(layers.Layer):
    def __init__(self,units = 32,**kwargs):
        super(Net,self).__init__(**kwargs)
        self.activity_regularizer = layers.ActivityRegularization(0.,1.)#规则化
        with tf.name_scope('layer1'):
            self.hiddenLayer = LinearLayer(units=10)#10是中间维度
        self.dropoutLayer = layers.Dropout(rate=PROB)
        with tf.name_scope('layer2'):
            self.outputLayer = LinearLayer(units=units)


    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs,step):#不需要加激励函数，因为每一层Linear都带有激励函数
        inputs = self.activity_regularizer(inputs)

        self.hiddenLayer.step = step#有点累赘，怎么解决？？

        hidden_output = self.hiddenLayer(inputs,step)
        # hidden_output = self.dropoutLayer(hidden_output)

        self.outputLayer.step = step#

        outputs = self.outputLayer(hidden_output,step)

        return outputs
        

def main():
    with tf.name_scope('input'):
        x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
        noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
        y_data = np.square(x_data) - 0.5 + noise#np.square不要写成tf.square

    net = Net(1)

    optimizer = optimizers.Adam(learning_rate=LR,beta_1=0.9,beta_2=0.99)
    # optimizer = keras.optimizers.SGD(learning_rate=LR)

    loss_fn = losses.MeanSquaredError()
    # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)#用这个函数会出错

    _,ax = plt.subplots(1,1,figsize=(8,6))

    for epoch in range(EPOCH):
        ax.cla()
        with tf.GradientTape() as tape:
            y_prediction = net(x_data,step=epoch)
            with tf.name_scope('loss'):
                loss = loss_fn(y_data,y_prediction)
                with writer.as_default():
                    tf.summary.scalar('loss',loss,step=epoch)
        ax.scatter(x_data,y_data,c='r')
        ax.plot(x_data,y_prediction,color='g')
        plt.pause(0.01)
        grads = tape.gradient(loss,net.trainable_weights)

        optimizer.apply_gradients(zip(grads,net.trainable_weights))
        # train_step = optimizer.minimize(loss,var_list=net.trainable_weights,tape=tf.GradientTape())#梯度加到需要更新的参数

        if epoch % 20 == 0:
            print(loss)

    plt.show()

    
if __name__=="__main__":
    main()
