### CS744 Assignment 2

[Problem Set](http://pages.cs.wisc.edu/~akella/CS744/S19/assignment2_html/assignment2.html)

### Part 1: Logistic Regression
### Task 1
Change to `LogisticRegression` directory
* Implemet the LR application and train the model using single node mode.
Code: Present in file `code_template_part_1.py` in the `LogisticRegression`

    Run using bash script:
```
    ./run_lr_singlenode.sh
```
Additionally if you want to play around by giving customized values for batch_size, learning rate, number od epochs. You may directly run:
 ```
    bash run_code_template.sh code_template_part_1.py single 100 10 0.01
    bash run_code_template.sh code_template_part_1.py <DEPLOY_MODE> <BATCH_SIZE> <N_EPOCHS> <LEARNING_RATE>
 ```

### Task 2
* Implemet the LR application in distributed mode using Sync SGD and Async SGD

#### Async SGD
Code: Present in file `code_template_asynch.py` in the `LogisticRegression`
 
Run using bash script:
```
    ./run_lr_async_cluster.sh
```

#### Sync SGD
Code: Present in file `code_template_synch_monitor.py` in the `LogisticRegression`

Run using bash script:
```
    ./run_lr_sync_cluster.sh
```

By default, the bash script for Sync and Async SGD trains it on cluster of 2 workers with batch size set as `100`, learning rate as `0.01` and number of epochs as `10`.

### Task 3
Try different batch size and see the difference.
For both Async and Sync SGD training, we customize to take any variable number of batches and did the comparison on the same. We tried with batch sizes 10, 50, 100, 200 and 500.

Run using bash script:
```
    bash run_code_template.sh code_template_asynch.py <DEPLOY_MODE> <BATCH_SIZE> <N_EPOCHS> <LEARNING_RATE>
```
where `<DEPLOY_MODE>` can be `single` (Single node cluster), `cluster` (Cluster of 2 workers), `cluster2` (Cluster of 3 workers) , the config of which is defined in the respective `*.py` files.

### Part 2: AlexNet
### Task 1
Change to `AlexNetCNN` directory


* Redo the task 2 from Part 1 using AlexNet in sync mode only. You can use the given optimizer instead of SGD.

Code: Present in file `alexnetmodes.py` in the `AlexNet/nets/` directory
First run the servers using bash script:
```
    ./startservers.sh <DEPLOY_MODE>
```
where deploy mode could be `single`, `cluster` or  `cluster2`
    
Run the program using the command:
```
    python -m AlexNet.scripts.train --mode cluster --batch_size 128
```
Ensure that the <DEPLOY_MODE> in starting the tensorflow service is same as while running the job.

For task 1, we run it using the deploy mode as `cluster`.


### Task 2
* Run the AlexNet using two machines. Monitor the CPU/Memory/Network usage and compare it to the three machine scenario. Remember you will need to modify startservers.sh to run in the correct mode.

Run the program using the command:
```
    python -m AlexNet.scripts.train --mode <DEPLOY_MODE> --batch_size <BATCH_SIZE>
```
where you can change `<DEPLOY_MODE>` to `cluster` (Cluster of 2 workers) or `cluster2` (Cluster of 3 workers) for the comparison.

For testing on different batches, the batch size can be changed.


