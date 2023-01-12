Our Environment：Python 3.8 + TensorFlow 2.6.0
Dataset: REDS video dataset
Setup Step:
1. Download REDS dataset from official web-site;
2. Install TensorFlow 2.6.0 (GPU version);
3. Check the path in dataset.py;
4. Run ea_test.py to test and start the evolution process;
4.1 eg. $python -u ea_test.py
      We recommend using the nohup command:
      eg. $nohup python -u ea_test.py > ea.out &
5. Run train_worker_tf.py to initiate and start train workers;
5.1 Get help: $python -u train_worker_tf.py -h
      We set the unique name of worker as unique worker id. Only the worker with same name can restart the previous train process.
6. The database will be generated automatically and models will be saved automatically.
6.1 You can use Sqlite3 related tools to access database data.