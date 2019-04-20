import tensorflow as tf
import os
import numpy as np
import time
import datetime
import sys

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
tf.app.flags.DEFINE_string("batch_size", "100", "either single or cluster")
tf.app.flags.DEFINE_string("n_epochs", "6", "either single or cluster")
tf.app.flags.DEFINE_string("learning_rate", "0.01", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                  stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2232"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
    ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
    ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	print(FLAGS.learning_rate)
      
	# learning_rate = float(FLAGS.learning_rate)
	batch_size = int(FLAGS.batch_size)
	learning_rate = float(FLAGS.learning_rate)
	n_epochs = int(FLAGS.n_epochs)
	
	n_features = 784
	n_classes = 10

	x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
	y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
	w = tf.Variable(tf.zeros([n_features, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))

	prediction = tf.nn.softmax(tf.matmul(x, w) + b)
	loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))
	
	

	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     
	tf.summary.histogram("prediction", prediction)
	# variable_summaries(prediction)
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("accuracy", accuracy)
	tf_summary = tf.summary.merge_all()

	optimizer = tf.train.GradientDescentOptimizer
	optimizer_f = optimizer(learning_rate=learning_rate).minimize(loss)


	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	train_writer = tf.summary.FileWriter("log/part1_%s" %(FLAGS.deploy_mode) , sess.graph)
	n_batches = int(mnist.train.num_examples/batch_size)
	print("n_batches %d" %(n_batches))
	start_time = datetime.datetime.now()
	for epoch in range(n_epochs):    
		for batch in range(n_batches):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			_, merged_summary= sess.run([optimizer_f, tf_summary], feed_dict={x: batch_xs, y: batch_ys}) 
			# print("At time %s, epoch %d, batch %d" %(str(datetime.datetime.now()),epoch,batch))
			train_writer.add_summary(merged_summary, n_batches*epoch + batch)
		current_time = datetime.datetime.now()
		print("Epoch done:%d, accuracy: %s, time: %s" %(epoch, sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}), str((current_time-start_time).total_seconds())))
	end_time = datetime.datetime.now()
	print("Total_time_taken:%s" %(str((end_time-start_time).total_seconds())))
	print("process done")


