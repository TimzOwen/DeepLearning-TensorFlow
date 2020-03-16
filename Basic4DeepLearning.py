#install and import tensorflow
pip install tensorflow
pip install keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Create graphs and get some operations in it
graph=tf.compat.v1.get_default_graph()
graph.get_operations()
for op in graph.get_operations():
    print(op.name)

#Sessions in TF;
#graphs only run when sessions are incoperated the they have no variables/values in it.
sess = tf.Session()
        #Your code
        #Your Code ..        
    sess.close()
    
#or use the block way to automatically close
with tf.Session as sess():
    sess.run(f)
    
#Constants:
a=tf.constant(1.0)
a
<tf.Tensor'Const:0' shape=() dtype=float32>
print(a)
Tensor("Const:0", shape=(), dtype=float32)
with tf.Session() as sess:
    print(sess.run(a))

#constants in tf
a = tf.constant(5.0)
with tf.Session() as sess:
    print(sess.run(a))

#varsibales / takes in data and can change
b = tf.Variable(2.0, name="test_var")
b
tensorflow.python.ops.variables.Variable object at 0x7f37ebda1990

#initializers for both versions
init_op = tf.global_variables_initializer() #greater > version
init_op = tf.initialize_all_variables() #<version

#Now running this outputs
graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)
    
        #     Const
        # test_var/initial_value
        # test_var
        # test_var/Assign
        # test_var/read
        # init

#PLACEHOLDERS
#this as tensors awaiting initializations and used esp during training of a model
#when sessions are running
#feed_dict are data fed to placeholders
a=tf.placeholder("float")
b=tf.placeholder("float")
y=tf.multiply(a,b)
feed_dict={a:2,b:3}
with tf.Session() as sess:
    print(sess.run(y,feed_dict))

#TF on Devices;
#tensorflow allows you to run your code on GPU and CPU due to its inbuilt capabilities


#TENSORFLOW SIMPLE PRACTICE FROM THE BASIC ABOVE.
#we are going to use some basic math functions in tensorflow
#1.0 = RANDOM NORMAL DISTRIBUTION
#random is used to create random values with standard deviation values
w=tf.Variable(tf.random_normal([20, 5], stddev=5))
#gives
[[ -2.0831275    5.2632422    5.27458      0.7123264    3.0437565 ]
 [ -1.1948549    3.50932      1.1581041   -4.1190686   -9.482818  ]
 [ -0.82041526  -6.3814154   -3.5882666   -7.624322     0.80383706]
 [  0.22200711  -6.8508863   -2.4307878   -1.9598747   -8.767801  ]
 [  6.8670626   -9.539546     2.0181117   -0.75715303   2.030138  ]
 [ -0.59764135   5.5903454   -1.4198912    6.1212707    4.0948257 ]
 [-11.787212     2.7954645   -0.3828703    1.2754753    4.760849  ]
 [  7.3233376    1.6561736    0.10633609  -4.7937036  -14.042471  ]
 [ -5.3090935   -8.701149    -5.6267633   -1.1451317    3.3174005 ]
 [  2.6073537    1.9067699    2.9175496   -0.33022207  -0.36176863]
 [  1.6869953   -0.9946976    3.4746714   -0.8411128    3.0976727 ]
 [  0.11553674  -0.546666    -3.8259063   -5.8025355   -6.286325  ]
 [ -2.7737916   -6.759117     0.7769965   -3.1598415   -6.599145  ]
 [  3.8853998    0.7108902    4.0805383   -6.6151447    2.6734276 ]
 [ 11.761485    -5.7045307   -3.4272223    2.8105874    1.7879785 ]
 [  2.7991924    5.074051     1.2459023    2.407752    -2.4353285 ]
 [  5.8895545    4.117129    12.655201    -2.9868052    8.475735  ]
 [  4.7934556   -5.021701   -11.623683    -0.17030223  -2.1690817 ]
 [ 14.057177     2.9639053  -11.425516     1.5357378   -6.434236  ]
 [ -5.7681637   -2.8020537   -2.3116841    0.41358596  -4.1856446 ]]

#2.0
w=tf.Variable(tf.random_normal([784, 10], stddev=0.01))
print(w)
#gives
<tf.Variable 'Variable_18:0' shape=(784, 10) dtype=float32_ref>

#3.0
w=tf.Variable(tf.random_normal([784, 10], stddev=0.01))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))
#gives
[[ 0.02092924  0.01622685  0.00950744 ... -0.00077061 -0.00364478
  -0.01419592]
 [ 0.01073358 -0.00338707  0.00842206 ... -0.00109519  0.0049654
  -0.00247096]
 [ 0.00334247 -0.01098258 -0.01388305 ... -0.00318701  0.00255408
  -0.00361468]
 ...
 [-0.0021882  -0.01192918 -0.00902387 ...  0.00484873  0.00335111
   0.00315019]
 [ 0.00458188  0.00130178  0.00781    ... -0.00417302 -0.01217698
   0.01474298]
 [ 0.00287902 -0.01055623 -0.01555232 ...  0.00241112  0.00798495
   0.01077066]]

#REDUCED MEAN.
#calculating mean of an array
mean=tf.Variable([10,20,30,40,50,60],name='t')
with tf.Session() as sess:
    sess.run(global_variables_initializer())
    print(sess.run(reduce_mean(mean)))
#gives
35

#ARGMAX
#this gets you a maximum value along a specified axis in a graph
a=[ [0.1, 0.2,  0.3  ],
    [20,  2,       3   ]
  ]
b = tf.Variable(a,name='b')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(tf.argmax(b,1)))
#gives 
2,0

a = tf.Variable([[2,1,0.5,1.6],[2,4,6,8,]])
b = tf.Variable(a,name='b')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.argmax(b,1)))
#gives 
[1,3]

#LINEAR REGRESSION 
#try fit 100 datasets into one line
