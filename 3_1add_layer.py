from unicodedata import name
import tensorflow._api.v2.compat.v1 as tf#加上这两句才能在tensorflow2.x运行1.x
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

PROB = 1

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None) -> tf.Variable:
    layer_name = 'layer{}'.format(n_layer)#add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))#注意:是random_normal
            tf.summary.histogram(layer_name+'/weights',Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+'/biases',biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases#inputs和Weights不要混了

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram(layer_name+'/outputs',outputs)

    return outputs

def main()->np.void:
    #构建数据
    x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise#np.square不要写成tf.square


    #定义网络搭好计算图，1个输入层，10个隐藏层，1个输出层,此时需要占位符来占计算图的位置
    with tf.name_scope('input'):#用input大图层包含
        xs = tf.placeholder(dtype=np.float32,shape=[None, 1],name='x_in')
        ys = tf.placeholder(dtype=np.float32,shape=[None,1],name='y_in')

    with tf.name_scope('layer2'):
        keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

    hidden_output = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
    hidden_output = tf.nn.dropout(hidden_output,keep_prob)
    prediction = add_layer(hidden_output,10,1,n_layer=1)

    #将计算loss(均方损失)的步骤（节点）接入计算图
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))#[1]表示标量?
        tf.summary.scalar('loss',loss)

    with tf.name_scope('train'):
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        train_step = tf.train.AdamOptimizer(0.1,0.9,0.999).minimize(loss)


    #初始化变量，才能使用数据
    init = tf.global_variables_initializer()
    #merge所有可视化
    merged = tf.summary.merge_all()
    #执行init的初始化步骤
    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter("./graph/logs1/", sess.graph)


    #网络搭建好了，开始训练
    _,ax = plt.subplots(1,1,figsize=(8,6))
    # plt.ion()#本次运行请注释，全局运行不要注释
    

    for i in range(100):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data,keep_prob:[PROB]})
        ax.cla()
        prediction_value = sess.run(prediction,feed_dict={xs:x_data,keep_prob:[PROB]})#不需要ys:y_data，因为不需要用到
        ax.scatter(x_data,y_data,c='r')
        ax.plot(x_data,prediction_value,color='g')#不要和上面的变量重名
        plt.pause(0.01)
        
        if i % 20 == 0:
            rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data,keep_prob:[PROB]})
            writer.add_summary(rs,i)
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data,keep_prob:[PROB]}))#每次都要feed_dict占位符

    plt.show()

if __name__ == '__main__':
    main()