import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100  # go through 100 features at a time (to save memory


x = tf.placeholder('float') #is a 28 by 28, so we smashed (flattened) the array for 784 inputs. Good for error checking
y = tf.placeholder('float')            #labels

def neural_network_model(my_data):
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl1],stddev=0.1)),'biases':tf.Variable(tf.ones(n_nodes_hl1))} #Will generate a bunch of random weigths in a giant random tensor
    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1)),'biases':tf.Variable(tf.ones(n_nodes_hl2))} 
    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3],stddev=0.1)),'biases':tf.Variable(tf.ones(n_nodes_hl3))}
    output_layer   = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes],stddev=0.1)),'biases':tf.Variable(tf.ones([n_classes]))}

    # (input_data*weights)+biases
    
    #enters through L1
    
    l1 = tf.add(tf.matmul(my_data, hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    
    #enters through L2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)
    
    #enters through L3
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)
    
    #gets to the output
    output = tf.matmul(l3, output_layer['weights'] + output_layer['biases'])

    return output

def train_nn(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(cost)
    
    #Cycles of feed fwd + backprop
    how_many_epochs = 20
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(how_many_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, 'completed out of ', how_many_epochs, ' loss ',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))    #COmpare result of NN to the test labels
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
        
train_nn(x)