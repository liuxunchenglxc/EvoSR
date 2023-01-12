import tensorflow as tf
import pathlib


def lite_convert(saved_model_dir, tflite_model_save_dir, model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path(tflite_model_save_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    model_filename = model_name + ".tflite"
    model_f16_filename = model_name + "_f16.tflite"
    tflite_model_file = tflite_models_dir / model_filename
    tflite_model_file.write_bytes(tflite_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter.convert()
    tflite_model_fp16_file = tflite_models_dir / model_f16_filename
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    lite_convert(saved_model_dir='./save/xxx',
                 tflite_model_save_dir='tflite_models',
                 model_name='xxx')
