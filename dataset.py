import tensorflow as tf


def get_reds_train_dset():
    seed = 666666
    label_path_ds = tf.data.Dataset.list_files("/data/REDS/train/train_sharp/*/*5.png", True, seed)
    image_path_ds = tf.data.Dataset.list_files("/data/REDS/train/train_sharp_bicubic/X4/*/*5.png", True, seed)
    ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

    def preprocess_image(image):
        image = tf.cast(tf.image.decode_png(image, channels=3), tf.float32)
        # image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    def load_and_preprocess_from_path_label(image_path, label_path):
        return load_and_preprocess_image(image_path), load_and_preprocess_image(label_path)

    image_label_ds = ds.map(load_and_preprocess_from_path_label, num_parallel_calls=4)
    return image_label_ds, 24000/10


def get_reds_val_dset():
    seed = 666666
    label_path_ds = tf.data.Dataset.list_files("/data/REDS/val/val_sharp/*/*5.png", True, seed)
    image_path_ds = tf.data.Dataset.list_files("/data/REDS/val/val_sharp_bicubic/X4/*/*5.png", True, seed)
    ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

    def preprocess_image(image):
        image = tf.cast(tf.image.decode_png(image, channels=3), tf.float32)
        # image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    def load_and_preprocess_from_path_label(image_path, label_path):
        return load_and_preprocess_image(image_path), load_and_preprocess_image(label_path)

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    return image_label_ds, 3000/10


def get_reds_train_dset_all():
    seed = 666666
    label_path_ds = tf.data.Dataset.list_files("/data/REDS/train/train_sharp/*/*.png", True, seed)
    image_path_ds = tf.data.Dataset.list_files("/data/REDS/train/train_sharp_bicubic/X4/*/*.png", True, seed)
    ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

    def preprocess_image(image):
        image = tf.cast(tf.image.decode_png(image, channels=3), tf.float32)
        # image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    def load_and_preprocess_from_path_label(image_path, label_path):
        return load_and_preprocess_image(image_path), load_and_preprocess_image(label_path)

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    return image_label_ds, 24000


def get_reds_val_dset_all():
    seed = 666666
    label_path_ds = tf.data.Dataset.list_files("/data/REDS/val/val_sharp/*/*.png", True, seed)
    image_path_ds = tf.data.Dataset.list_files("/data/REDS/val/val_sharp_bicubic/X4/*/*.png", True, seed)
    ds = tf.data.Dataset.zip((image_path_ds, label_path_ds))

    def preprocess_image(image):
        image = tf.cast(tf.image.decode_png(image, channels=3), tf.float32)
        # image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    def load_and_preprocess_from_path_label(image_path, label_path):
        return load_and_preprocess_image(image_path), load_and_preprocess_image(label_path)

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    return image_label_ds, 3000