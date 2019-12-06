import os
import random

"""
get a pair of photot and indicate wheter they are the same person or not 
"""
class PairGenerator(object):
    def __init__(self, lfw_path='recognizer'+ os.sep + 'resources' + os.sep + 'lfw'):
        self.all_people = self.generate_all_people_dict(lfw_path)
 
    def generate_all_people_dict(self, lfw_path):
    # generate a dictionary between a person and all the photos of that person
        all_people = dict()
        for person_folder in os.listdir(lfw_path):
            person_photos = glob.glob(lfw_path + os.path.sep + persion_folder + os.path.sep + '*.jpg')
            all_people[person_folder] = person_photos
        return all_people

    def get_next_pair(self):
        while True:
            # draw a person at random
            person1 = random.choice(self.all_people)
            # flip a coin to decide wheteher we fetch a photo of the same person vs different person
            same_person = random.random() > 0.5
            if same_person :
                person2 = person1
            else:
                person2 = random.choice(self.all_people)
            person1_photo = random.choice(self.all_people[person1])
            person2_photo = random.choice(self.all_people[person2])
            yield ({'person1': person1_photo, 
                    'person2': person2_photo, 
                    'label': same_person})
