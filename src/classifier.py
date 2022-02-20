import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np


class Classifier:
    def __init__(self):
        self.model = tf.keras.models.load_model('../flowers-nn')
        self.model.summary()
        self.imgSize = (180, 180)
        # hardwired to avoid reloading training set
        self.classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    def classify(self, image):
        PILImage = Image.open(BytesIO(image))

        imgAsArray = tf.keras.utils.img_to_array(PILImage)
        imgAsArray = tf.keras.preprocessing.image.smart_resize(
            imgAsArray, self.imgSize)
        imgAsArray = tf.expand_dims(imgAsArray, 0)  # Create a batch

        predictions = self.model.predict(imgAsArray)
        score = tf.nn.softmax(predictions[0])

        return {'species': self.classes[np.argmax(score)],
                'confidence': 100 * np.max(score)}
