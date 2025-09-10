import os
import shutil
import dataset
import tensorflow as tf


def train(gpu_idx=0, model_class=None, model_ckpt_dir="./tf_cpkt",
          model_save_dir="./tf_save", log_dir="./log/log",
          epochs=5, batch_size=4):
    if gpu_idx >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
    train_log_dir = log_dir + '/train'
    val_log_dir = log_dir + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    with tf.device('/device:GPU:0'):
        model = model_class()
    loss_object1 = tf.keras.losses.MeanAbsoluteError()
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(1e-3 * 16, 200, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_psnr = tf.keras.metrics.Mean(name='train_psnr')
    train_ssim = tf.keras.metrics.Mean(name='train_ssim')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_psnr = tf.keras.metrics.Mean(name='val_psnr')
    val_ssim = tf.keras.metrics.Mean(name='val_ssim')
    train_ds, train_ds_len = dataset.get_reds_train_dset()
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=2)
    val_ds, val_ds_len = dataset.get_reds_val_dset()
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=2)
    train_ds_cache = [i for i in iter(train_ds)]
    val_ds_cache = [i for i in iter(val_ds)]

    @tf.function
    def train_step(images, labels):
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss1 = loss_object1(labels, predictions)
                gradients = tape.gradient(loss1, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss1)
                train_psnr(tf.image.psnr(labels, predictions, max_val=255))  # for int8
                train_ssim(tf.image.ssim(labels, predictions, max_val=255))  # for int8

    @tf.function
    def val_step(images, labels):
        with tf.device('/device:GPU:0'):
            predictions = model(images)
            loss1 = loss_object1(labels, predictions)

            val_loss(loss1)
            val_psnr(tf.image.psnr(labels, predictions, max_val=255))  # for int8
            val_ssim(tf.image.ssim(labels, predictions, max_val=255))  # for int8

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=optimizer, model=model, best_psnr=tf.Variable(0.0),
                               the_ssim=tf.Variable(0.0))
    manager = tf.train.CheckpointManager(ckpt, model_ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    for epoch in range(epochs - ckpt.epoch):
        train_loss.reset_states()
        train_psnr.reset_states()
        val_loss.reset_states()
        val_psnr.reset_states()

        for images, labels in train_ds_cache:
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('psnr', train_psnr.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('ssim', train_ssim.result(), step=tf.cast(ckpt.epoch, tf.int64))

        for val_images, val_labels in val_ds_cache:
            val_step(val_images, val_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('psnr', val_psnr.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('ssim', val_ssim.result(), step=tf.cast(ckpt.epoch, tf.int64))

        template = 'Epoch {}, MAE: {}, PSNR: {}, SSIM: {}, Val MAE: {}, Val PSNR: {}, Val SSIM: {}'
        if val_psnr.result() > ckpt.best_psnr:
            try:
                os.removedirs(model_save_dir + "/best_temp")
            except:
                pass
            model.save(model_save_dir + "/best_temp")
            ckpt.best_psnr.assign(val_psnr.result())
            ckpt.the_ssim.assign(val_ssim.result())
        ckpt.epoch.assign_add(1)
        manager.save()
        print(template.format(int(ckpt.epoch),
                              train_loss.result(),
                              train_psnr.result(),
                              train_ssim.result(),
                              val_loss.result(),
                              val_psnr.result(),
                              val_ssim.result()))
    final_model_dirname = "best_" + '{}'.format(tf.cast(ckpt.best_psnr * 1e6, tf.int32))
    final_model_path = model_save_dir + "/" + final_model_dirname
    shutil.move(model_save_dir + "/best_temp", final_model_path)
    return final_model_dirname, float(ckpt.best_psnr), float(ckpt.the_ssim), 0


def train_second(gpu_idx=0, model_class=None, model_ckpt_dir="./tf_cpkt",
          model_save_dir="./tf_save", log_dir="./log/log",
          epochs=5, batch_size=4, lr=1e-3 * 16):
    if gpu_idx >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
    train_log_dir = log_dir + '/train'
    val_log_dir = log_dir + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    with tf.device('/device:GPU:0'):
        model = model_class()
    loss_object1 = tf.keras.losses.MeanAbsoluteError()
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(lr, 200, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_psnr = tf.keras.metrics.Mean(name='train_psnr')
    train_ssim = tf.keras.metrics.Mean(name='train_ssim')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_psnr = tf.keras.metrics.Mean(name='val_psnr')
    val_ssim = tf.keras.metrics.Mean(name='val_ssim')
    train_ds, train_ds_len = dataset.get_reds_train_dset_all()
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=2)
    val_ds, val_ds_len = dataset.get_reds_val_dset_all()
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=2)

    @tf.function
    def train_step(images, labels):
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss1 = loss_object1(labels, predictions)
                gradients = tape.gradient(loss1, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss1)
                train_psnr(tf.image.psnr(labels, predictions, max_val=255))  # for int8
                train_ssim(tf.image.ssim(labels, predictions, max_val=255))  # for int8

    @tf.function
    def val_step(images, labels):
        with tf.device('/device:GPU:0'):
            predictions = model(images)
            loss1 = loss_object1(labels, predictions)

            val_loss(loss1)
            val_psnr(tf.image.psnr(labels, predictions, max_val=255))  # for int8
            val_ssim(tf.image.ssim(labels, predictions, max_val=255))  # for int8

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=optimizer, model=model, best_psnr=tf.Variable(0.0),
                               the_ssim=tf.Variable(0.0))
    manager = tf.train.CheckpointManager(ckpt, model_ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    for epoch in range(epochs - ckpt.epoch):
        train_loss.reset_states()
        train_psnr.reset_states()
        val_loss.reset_states()
        val_psnr.reset_states()

        for images, labels in iter(train_ds):
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('psnr', train_psnr.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('ssim', train_ssim.result(), step=tf.cast(ckpt.epoch, tf.int64))

        for val_images, val_labels in iter(val_ds):
            val_step(val_images, val_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('psnr', val_psnr.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('ssim', val_ssim.result(), step=tf.cast(ckpt.epoch, tf.int64))

        template = 'Epoch {}, MAE: {}, PSNR: {}, SSIM: {}, Val MAE: {}, Val PSNR: {}, Val SSIM: {}'
        if val_psnr.result() > ckpt.best_psnr:
            try:
                os.removedirs(model_save_dir + "/best_temp")
            except:
                pass
            model.save(model_save_dir + "/best_temp")
            ckpt.best_psnr.assign(val_psnr.result())
            ckpt.the_ssim.assign(val_ssim.result())
        ckpt.epoch.assign_add(1)
        manager.save()
        print(template.format(int(ckpt.epoch),
                              train_loss.result(),
                              train_psnr.result(),
                              train_ssim.result(),
                              val_loss.result(),
                              val_psnr.result(),
                              val_ssim.result()))
    final_model_dirname = "best_" + '{}'.format(tf.cast(ckpt.best_psnr * 1e6, tf.int32))
    final_model_path = model_save_dir + "/" + final_model_dirname
    shutil.move(model_save_dir + "/best_temp", final_model_path)
    return final_model_dirname, float(ckpt.best_psnr), float(ckpt.the_ssim), 0
