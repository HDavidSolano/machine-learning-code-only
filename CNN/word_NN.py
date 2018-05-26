import tensorflow as tf
import numpy as np
import pickle

train_x,train_y,test_x,test_y = pickle.load( open( "sentimet_set.pickle", "rb" ) )

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 200  # go through 100 features at a time (to save memory)


x = tf.placeholder('float',[None,len(train_x[0])]) #array of inputs for neural net
y = tf.placeholder('float')            #labels

def neural_network_model(my_data):
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1],stddev=0.1)),'biases':tf.Variable(tf.ones(n_nodes_hl1))} #Will generate a bunch of random weigths in a giant random tensor
    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2],stddev=0.1)),'biases':tf.Variable(tf.ones(n_nodes_hl2))} 
    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3],stddev=0.1)),'biases':tf.Variable(tf.ones(n_nodes_hl3))}
    output_layer   = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes],stddev=0.1)),'biases':tf.Variable(tf.ones([n_classes]))}

    # (input_data*weights)+biases
    
    #enters through L1
    
    l1 = tf.add(tf.matmul(my_data, hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    #enters through L2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    #enters through L3
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    #gets to the output
    output = tf.matmul(l3, output_layer['weights'] + output_layer['biases'])

    return output

def train_nn(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #Cycles of feed fwd + backprop
    how_many_epochs = 20
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(how_many_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, c = sess.run([optimizer,cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch ', epoch, 'completed out of ', how_many_epochs, ' loss ',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))    #COmpare result of NN to the test labels
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
        print('Accuracy',accuracy.eval({x:test_x,y:test_y}))
        
train_nn(x)


