from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf

class ResNet50Dog:

    def __init__(self):
        pass

    def build(self):
        # https://github.com/keras-team/keras/issues/6462
        self.resNet50_model = ResNet50(weights='imagenet')
        self.graph = tf.get_default_graph()

    def predict_labels(self, features):
        # returns prediction vector for image located at img_path
        img = preprocess_input(features)
        with self.graph.as_default():
            return np.argmax(self.resNet50_model.predict(img))

    def is_dog(self, features):
        prediction = self.predict_labels(features)
        return ((prediction <= 268) & (prediction >= 151)) 