from .resnet50_dog import ResNet50Dog
from .dog_classifier import DogClassifier
from enum import Enum

import os 
from keras.preprocessing import image   
import numpy as np
import cv2
import getopt, sys

class ProcessCode(Enum):
    HUMAN_FACE = 1
    DOG = 2
    NEITHER = 3

class DogIdentification:

    def __init__(self):
        pass

    def build(self):
        # Build all required networks

        # Face Detector
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(package_directory, 'haarcascades', 'haarcascade_frontalface_alt.xml')
            )

        # Dog Detector 
        self.resnet50_dog = ResNet50Dog()
        self.resnet50_dog.build()
        
        # Dog Classifier based on Xception
        self.dog_classifier = DogClassifier()
        self.dog_classifier.build(
            os.path.join(package_directory, 'checkpoints', 'weights.best.Xception.hdf5')
            )
        
    def is_face_detected(self, image=None):
        self._check_features(image)
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def load_image(self, file_path):
        img = image.load_img(file_path, target_size=(224, 224))
        self.image_path = file_path
        self.features = np.zeros((1,224,224,3))
        self.features[0,:,:,:] = image.img_to_array(img)

    def dog_detector(self, image=None):
        self._check_features(image)
        return self.resnet50_dog.is_dog(self.features)

    def dog_classify(self, image=None): 
        self._check_features(image)
        return self.dog_classifier.predict(self.features)

    # Check if human or dog 
    #   - if human: verify if face is clearly visible & return dog breed most resembling 
    #   - if dog: provide an estimate of the dogs breed 
    # 
    # Returns:
    #   - status
    #   - information string
    #   - dictionary with probabilities & most likely breeds 
    def process(self, image=None):
        self._check_features(image)
        # check if dog 
        if self.dog_detector():
            code = ProcessCode.DOG
        else:
            # if not we can expect the image to be of a human 
            # check if face is clearly identifiable
            if self.is_face_detected():
                code = ProcessCode.HUMAN_FACE
            else:
                code = ProcessCode.NEITHER
                return (code, 'Neither a dog nor a human face' , {})

        # get predictions 
        predictions = self.dog_classify()
        if code == ProcessCode.HUMAN_FACE:
            info = 'Human face'
        else:
            info = 'Dog'

        return (code, info, predictions)

    def _check_features(self, image):
        if image != None:
            self.load_image(image)

        if type(self.features) == type(None):
            raise ValueError('Image not provided')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, '', ['file=', 'display='])
        disp = 5
        for o,a in opts:
            if o == '--file':
                file_path =  a
            elif o == '--display':
                disp = int(a)
    except getopt.GetoptError as err:
        print(err)

    if file_path != None:
        app = DogIdentification()
        app.build()
        code, info, predictions = app.process(file_path)
        print(info)
        if code == ProcessCode.NEITHER:
            return

        for i in range(disp):
            print("{},{:.3f}".format( predictions['breeds'][i], predictions['prob'][i] ))

    import gc; gc.collect()

package_directory = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    main(sys.argv[1:])
