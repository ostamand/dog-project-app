from keras.callbacks import ModelCheckpoint 
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.applications.xception import Xception, preprocess_input
import numpy as np
import tensorflow as tf

class DogClassifier():

    def __init__(self):
        pass

    def build(self, checkpoint):
        self.base_model = Xception(include_top=False, weights='imagenet')

        self.Xception_model = Sequential()
        self.Xception_model.add(GlobalAveragePooling2D(input_shape= (7, 7 , 2048) ))
        self.Xception_model.add(Dense(200, activation='tanh'))
        self.Xception_model.add(Dropout(0.7))
        self.Xception_model.add(Dense(133, activation='softmax'))

        self.Xception_model.load_weights(checkpoint)

        self.graph = tf.get_default_graph()

        self.dog_names = ['Affenpinscher',
        'Afghan hound',
        'Airedale terrier',
        'Akita',
        'Alaskan malamute',
        'American eskimo dog',
        'American foxhound',
        'American staffordshire terrier',
        'American water spaniel',
        'Anatolian shepherd dog',
        'Australian cattle dog',
        'Australian shepherd',
        'Australian terrier',
        'Basenji',
        'Basset hound',
        'Beagle',
        'Bearded collie',
        'Beauceron',
        'Bedlington terrier',
        'Belgian malinois',
        'Belgian sheepdog',
        'Belgian tervuren',
        'Bernese mountain dog',
        'Bichon frise',
        'Black and tan coonhound',
        'Black russian terrier',
        'Bloodhound',
        'Bluetick coonhound',
        'Border collie',
        'Border terrier',
        'Borzoi',
        'Boston terrier',
        'Bouvier des flandres',
        'Boxer',
        'Boykin spaniel',
        'Briard',
        'Brittany',
        'Brussels griffon',
        'Bull terrier',
        'Bulldog',
        'Bullmastiff',
        'Cairn terrier',
        'Canaan dog',
        'Cane corso',
        'Cardigan welsh corgi',
        'Cavalier king charles spaniel',
        'Chesapeake bay retriever',
        'Chihuahua',
        'Chinese crested',
        'Chinese shar-pei',
        'Chow chow',
        'Clumber spaniel',
        'Cocker spaniel',
        'Collie',
        'Curly-coated retriever',
        'Dachshund',
        'Dalmatian',
        'Dandie dinmont terrier',
        'Doberman pinscher',
        'Dogue de bordeaux',
        'English cocker spaniel',
        'English setter',
        'English springer spaniel',
        'English toy spaniel',
        'Entlebucher mountain dog',
        'Field spaniel',
        'Finnish spitz',
        'Flat-coated retriever',
        'French bulldog',
        'German pinscher',
        'German shepherd dog',
        'German shorthaired pointer',
        'German wirehaired pointer',
        'Giant schnauzer',
        'Glen of imaal terrier',
        'Golden retriever',
        'Gordon setter',
        'Great dane',
        'Great pyrenees',
        'Greater swiss mountain dog',
        'Greyhound',
        'Havanese',
        'Ibizan hound',
        'Icelandic sheepdog',
        'Irish red and white setter',
        'Irish setter',
        'Irish terrier',
        'Irish water spaniel',
        'Irish wolfhound',
        'Italian greyhound',
        'Japanese chin',
        'Keeshond',
        'Kerry blue terrier',
        'Komondor',
        'Kuvasz',
        'Labrador retriever',
        'Lakeland terrier',
        'Leonberger',
        'Lhasa apso',
        'Lowchen',
        'Maltese',
        'Manchester terrier',
        'Mastiff',
        'Miniature schnauzer',
        'Neapolitan mastiff',
        'Newfoundland',
        'Norfolk terrier',
        'Norwegian buhund',
        'Norwegian elkhound',
        'Norwegian lundehund',
        'Norwich terrier',
        'Nova scotia duck tolling retriever',
        'Old english sheepdog',
        'Otterhound',
        'Papillon',
        'Parson russell terrier',
        'Pekingese',
        'Pembroke welsh corgi',
        'Petit basset griffon vendeen',
        'Pharaoh hound',
        'Plott',
        'Pointer',
        'Pomeranian',
        'Poodle',
        'Portuguese water dog',
        'Saint bernard',
        'Silky terrier',
        'Smooth fox terrier',
        'Tibetan mastiff',
        'Welsh springer spaniel',
        'Wirehaired pointing griffon',
        'Xoloitzcuintli',
        'Yorkshire terrier']

    def predict(self, base_features):

        with self.graph.as_default():
            features_top = self.base_model.predict(preprocess_input(base_features))
            predictions = self.Xception_model.predict(features_top)

        ind = np.flipud( np.argsort(predictions[0]) )
        breeds = []
        result = {'prob': np.flipud( np.sort(predictions[0]) ), 'breeds': breeds }
        for i in ind:
            breeds.append(self.dog_names[i])

        return result