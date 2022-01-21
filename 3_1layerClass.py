import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras import layers


import numpy as np
import matplotlib.pyplot as plt

#超参数
PROB = 1
LR = 0.01
IN_SIZE = 1
OUT_SIZE = 1
EPOCH = 100

#定义网络结构（只有一层）
class Net(layers.Layer) :
    def __init__(self, units = 32, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.activityLayer = layers.ActivityRegularization(0,1)#规则化
        self.dropoutLayer = layers.Dropout(rate=PROB)
        
        self.lr = LR
        self.units = units

    def build(self, input_shape):#添加本层要学习的参数
        self.Weights = self.add_weight(
            shape=(input_shape[-1],self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.biases = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )

        return super().build(input_shape)

    def call(self,inputs):#类似forward
        outputs = tf.matmul(inputs,self.Weights)+self.biases
        outputs = activations.relu(outputs)
        outputs = self.dropoutLayer(outputs)

        losses = self.lr * tf.reduce_sum(inputs)#改？
        self.add_loss(losses)
        return outputs

    def get_config(self):
        config = super(Net,self).get_config()
        config.update({"units":self.units})

        return config


def main():
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
            y_prediction = net(x_data)
            loss = loss_fn(y_data,y_prediction)
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
        
