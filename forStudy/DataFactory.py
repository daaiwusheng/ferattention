# from torchlib.datasets import fer
# from pytvision.datasets.imageutl import dataProvide
from pytvision.transforms.rectutils import Rect

from .CKPDataProvider import *


def getroi():

    # pts = self.getladmarks()
    # minx = np.min(pts[:,0]); maxx = np.max(pts[:,0]);
    # miny = np.min(pts[:,1]); maxy = np.max(pts[:,1]);
    # box = [minx,miny,maxx,maxy]

    box = [0, 0, 48, 48]
    face_rc = Rect(box)
    return face_rc


class CKPDataset(dataProvide):
    classes = ['Neutral - NE', 'Happiness - HA', 'Surprise - SU', 'Sadness - SA', 'Anger - AN', 'Disgust - DI',
               'Fear - FR', 'Contempt - CO']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, train=True, transform=None):
        # classes = ['Neutral - NE', 'Anger - AN', 'Contempt - CO', 'Disgust - DI', 'Fear - FR', 'Happiness - HA',
        # 'Sadness - SA', 'Surprise - SU']
        self.toferp = [0, 4, 7, 5, 6, 1, 3, 2]
        self.train = train
        self.ckp_data_provider = CKPDataProvider()
        if train:
            self.images = self.ckp_data_provider.train_images  # image's file name
        else:
            self.images = self.ckp_data_provider.val_images  # image's file name

        self.index = 0  # set 0 at beginning

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if i >= len(self):
            print('current index :', i, flush=True)
            raise ValueError('Index outside range')
        self.index = i
        # image = np.array(self.data[i].reshape(self.imsize).transpose(1, 0), dtype=np.uint8)
        image = np.array(self.ckp_data_provider.dict_image_imgarray[self.images[i]], dtype=np.uint8)
        label = self.get_fer_label(self.images[i])
        return image, label

    def iden(self, i):
        return self.images[i]

    def getladmarks(self):
        i = self.index
        image = self.images[i]
        return self.ckp_data_provider.dict_image_landmarks[image]

    def get_fer_label(self, image_name):
        lab = self.ckp_data_provider.dict_image_lab[image_name]
        return self.toferp[lab]
