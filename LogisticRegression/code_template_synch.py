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
        "10.10.1.1:2222"
    ],
    "worker": [
        "10.10.1.1:2223",
        "10.10.1.2:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps": [
        "10.10.1.1:2222"
    ],
    "worker": [
        "10.10.1.1:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222"
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

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    is_chief = (FLAGS.task_index == 0)
    with tf.device(
        tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=clusterinfo)):

        batch_size = int(FLAGS.batch_size)
        learning_rate = float(FLAGS.learning_rate)
        n_epochs = int(FLAGS.n_epochs)
        n_features = 784
        n_classes = 10

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
      #   tf.summary.histogram("prediction", prediction)
      #   tf.summary.scalar("loss", loss)
      #   tf.summary.scalar("accuracy", accuracy)
      #   tf_summary = tf.summary.merge_all()
        train_op = optimizer1.minimize(loss, global_step=global_step)
        
        local_init_op = optimizer1.local_step_init_op
        if is_chief:
            local_init_op = optimizer1.chief_init_op
      
        ready_for_local_init_op = optimizer1.ready_for_local_init_op
        chief_queue_runner = optimizer1.get_chief_queue_runner()
        sync_init_op = optimizer1.get_init_tokens_op()

        init = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()


        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            recovery_wait_secs=1,
            global_step=global_step)

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

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)
        n_workers = len(vars(clusterinfo)['_cluster_spec']['worker'])
        num_curr_examples = (mnist.train.num_examples / n_workers)
        print('Number of Workers: ', n_workers)
        print('Current shrad size: ', num_curr_examples)

        if is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])
      #   train_writer = tf.summary.FileWriter("log/synch_%s" %(FLAGS.deploy_mode) , sess.graph)
        local_step = 0
        start_time = datetime.datetime.now()
        for epoch in range(n_epochs):
        
            n_batches = int(mnist.train.num_examples/batch_size)
            
            for batch in range(n_batches):
                #if batch % n_workers != FLAGS.task_index:
                #    print('skipping this batch ', batch)
                #    continue
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, step =  sess.run([train_op, global_step], feed_dict={x: batch_xs, y: batch_ys})
                local_step += 1
            if step >= n_epochs*n_batches:
                break

            #     train_writer.add_summary(merged_summary, n_batches*epoch + batch)
            #     print("At time: %s: Worker %d: training step %d, epoch %d, batch %d, (global step: %d), accuracy %f" %
                  # (str(datetime.datetime.now()), FLAGS.task_index, local_step, epoch, batch, step, sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
            current_time = datetime.datetime.now()
            print("Epoch done:%d, accuracy: %s, time: %s" %(epoch, sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}), str((current_time-start_time).total_seconds())))    
                    
            #     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

            
        end_time = datetime.datetime.now()
        print("Total_time_taken:%s" %(str((end_time-start_time).total_seconds())))
        print("process done")
