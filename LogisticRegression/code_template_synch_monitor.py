import tensorflow as tf
import os
import numpy as np
import tempfile
import time
import datetime

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
tf.app.flags.DEFINE_string("batch_size", "100", "either single or cluster")
tf.app.flags.DEFINE_string("n_epochs", "6", "either single or cluster")
tf.app.flags.DEFINE_string("learning_rate", "0.01", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker": [
        "10.10.1.1:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps": [
        "10.10.1.1:2262"
    ],
    "worker": [
        "10.10.1.1:2263",
        "10.10.1.2:2262"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps": [
        "10.10.1.1:2262"
    ],
    "worker": [
        "10.10.1.1:2263",
        "10.10.1.2:2262",
        "10.10.1.3:2262"
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.deploy_mode == "single":
    REPLICAS_TO_AGGREGATE = 1
    num_workers = REPLICAS_TO_AGGREGATE
if FLAGS.deploy_mode == "cluster":
    REPLICAS_TO_AGGREGATE = 2
    num_workers = REPLICAS_TO_AGGREGATE
if FLAGS.deploy_mode == "cluster2":
    REPLICAS_TO_AGGREGATE = 3
    num_workers = REPLICAS_TO_AGGREGATE

print(REPLICAS_TO_AGGREGATE)
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    from tensorflow.examples.tutorials.mnist import input_data
    batch_size = int(FLAGS.batch_size)
    learning_rate = float(FLAGS.learning_rate)
    n_epochs = int(FLAGS.n_epochs)
    n_features = 784
    n_classes = 10
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_batches = int(mnist.train.num_examples/batch_size)
    is_chief = (FLAGS.task_index == 0)
    print(type(mnist.train))  
    with tf.device(
        tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=clusterinfo)):

        

        x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
        y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
        w = tf.Variable(tf.zeros([n_features, n_classes]))
        b = tf.Variable(tf.zeros([n_classes]))
        
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        prediction = tf.nn.softmax(tf.matmul(x, w) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=1))
        

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        optimizer1 = tf.train.SyncReplicasOptimizer(optimizer,
            replicas_to_aggregate=REPLICAS_TO_AGGREGATE, total_num_replicas=num_workers)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_op = optimizer1.minimize(loss, global_step=global_step, aggregation_method = tf.AggregationMethod.ADD_N)
        
      #   local_init_op = optimizer1.local_step_init_op
      #   if is_chief:
      #       local_init_op = optimizer1.chief_init_op
      
      #   ready_for_local_init_op = optimizer1.ready_for_local_init_op
      #   chief_queue_runner = optimizer1.get_chief_queue_runner()
      #   sync_init_op = optimizer1.get_init_tokens_op()

      #   init = tf.global_variables_initializer()
      #   train_dir = tempfile.mkdtemp()

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
      #   if is_chief:
      #       print("Worker %d: Initializing session..." % FLAGS.task_index)
      #   else:
      #       print("Worker %d: Waiting for session to be initialized..." %
      #           FLAGS.task_index)

        sync_replicas_hook = optimizer1.make_session_run_hook(is_chief, num_tokens=0)
        stop_hook = tf.train.StopAtStepHook(last_step=n_epochs * n_batches)
        hooks = [sync_replicas_hook]
        sess = tf.train.MonitoredTrainingSession(master = server.target, 
            is_chief=is_chief,
            config=sess_config,
            hooks=hooks,
            stop_grace_period_secs=1)
        print("Worker %d: Session initialization complete." % FLAGS.task_index)
        n_workers = len(vars(clusterinfo)['_cluster_spec']['worker'])
        num_curr_examples = (mnist.train.num_examples / n_workers)
        print('Number of Workers: ', n_workers)
        print('Current shrad size: ', num_curr_examples)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)
        start_time = datetime.datetime.now()
        while True:
            local_step = 0              
            for epoch in range(n_epochs):
                for batch in range((n_batches / n_workers)*n_workers):
                    if batch % n_workers != FLAGS.task_index:
                         #print('skipping this batch ', batch)
                         continue
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, step =  sess.run([train_op, global_step], feed_dict={x: batch_xs, y: batch_ys})
                    local_step += 1
                  #  print("At time: %s: Worker %d: training step %d, epoch %d, batch %d, (global step: %d), accuracy %f" %
                  #    (str(datetime.datetime.now()), FLAGS.task_index, local_step, epoch, batch, step, sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
                  #   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

                current_time = datetime.datetime.now()
                print("Epoch done:%d, accuracy: %s, time: %s" %(epoch, sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}), str((current_time-start_time).total_seconds())))    
            break
        end_time = datetime.datetime.now()
        print("Total_time_taken:%s" %(str((end_time-start_time).total_seconds())))
        print("process done")
