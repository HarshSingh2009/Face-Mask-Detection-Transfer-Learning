import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image


class PredictionPipeline():
    def __init__(self) -> None:
        self.CLASS_NAMES = ['mask', 'no mask']
        self.IMG_SIZE = 224

    def predict(self, input_img):
        model = load_model('inceptionv3_model.h5')
        image = Image.open(input_img)
        # Changing the dtype to float32
        image = tf.cast(image, dtype=tf.float32)
        # Normalize the image data to [0, 1]
        image = image / 255.0
        input_tensor = tf.expand_dims(tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE]), axis=0)
        # Making Predictions
        try:
            y_probs = model.predict(input_tensor)
        except ValueError as err:
            return [[-1]], err
        else:
            return tf.round(y_probs), y_probs
