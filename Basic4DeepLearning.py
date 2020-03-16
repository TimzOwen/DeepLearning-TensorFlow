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

