import os
import numpy as np
from .Tools import *
from pytvision.datasets.imageutl import dataProvide
import cv2
import h5py
from .SingletonClass import *

# classes = ['Neutral - NE', 'Anger - AN', 'Contempt - CO', 'Disgust - DI', 'Fear - FR', 'Happiness - HA', 'Sadness - SA', 'Surprise - SU']
Neutral = 0
Anger = 1
Contempt = 2
Disgust = 3
Fear = 4
Happiness = 5
Sadness = 6
Surprise = 7

emotion_labs = [Neutral, Anger, Contempt, Disgust, Fear, Happiness, Sadness, Surprise]


class CKPDataProvider(object,metaclass=Singleton):

    def __init__(self, split_factor=0.8):
        # on Linux
        self.root_dir = '/home/steven/桌面/AICode/project_dataset/CK+/Emotion'
        self.r_dir_images = '/home/steven/桌面/AICode/project_dataset/CK+/cohn-kanade-images/'
        self.r_dir_landmarks = '/home/steven/桌面/AICode/project_dataset/CK+/Landmarks/'
        # on Mac
        # self.root_dir = '/Users/wangyu/Desktop/利兹上课资料/MSCP/dataset/CK+/Emotion/'
        # self.r_dir_images = '/Users/wangyu/Desktop/利兹上课资料/MSCP/dataset/CK+/cohn-kanade-images/'
        # self.r_dir_landmarks = '/Users/wangyu/Desktop/利兹上课资料/MSCP/dataset/CK+/Landmarks/'

        self.dirs_emotion_actors = self.get_dirs_emotion_actors()  # actor means the code like S506,
        self.key_sequence_with_labels = []  # 存储 key值 (如S506/002),按这个key就可以索引出标签,图片,landmarks,而不会乱掉
        self.dict_key_seq_labels = {}  # 字典,存储key值,value是标签值
        self.get_nessary_data()
        self.dict_lab_ikeys = {}  # 字典, k 为 标签, v 为, image的key (如S506/002),根据 dict_key_seq_labels 得到
        self.get_dict_lab_ikeys()
        self.num_frames = 2
        self.dict_lab_images = {}  # 字典,k 为标签, v为 image的文件名, 图片去最后 num_frames 帧
        self.get_dict_lab_images()
        self.dict_image_lab = {}  # 字典,k为image文件名, v 为 label
        self.get_dict_image_lab()
        self.dict_image_landmarks = {}  # k is image file name, v is landmarks (array,2 dims)
        self.get_dict_image_landmarks()
        self.dict_image_imgarray = {}  # k is image file name, v is image's data in array (490,640),(480,720),(480,640)
        self.get_dict_image_imgarray()
        self.train_images, self.val_images = self.prepare_tran_val_data(split_factor)
        self.test_data()

    def prepare_tran_val_data(self, split_factor):
        train_images = []
        val_images = []
        for lab, images in self.dict_lab_images.items():

            images_tmp = np.array(images)
            if lab == 0:
                images_tmp = images_tmp[0:70]
            np.random.shuffle(images_tmp)
            len_train = int(split_factor * len(images))
            train_images.extend(images_tmp[:len_train])
            val_images.extend(images_tmp[len_train:])

        train_images = np.array(train_images)
        val_images = np.array(val_images)
        np.random.shuffle(train_images)
        np.random.shuffle(val_images)
        return train_images, val_images

    def test_data(self):
        # for ima_name in self.train_images:
        #     print('lab: ', self.dict_image_lab[ima_name])
        for lab, images in self.dict_lab_images.items():
            print(lab, len(images))

    def get_dirs_emotion_actors(self):
        dirs_emotion_actors = []
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            dirs = outclude_hidden_dirs(dirs)
            dirs.sort()
            for name in dirs:
                # print(os.path.join(root, name))
                dirs_emotion_actors.append(os.path.join(root, name))
            break

        return dirs_emotion_actors

    def get_nessary_data(self):
        emotion_files = []  # 存储表情标签的txt 文件的路径
        lables = []
        for emotion_actor in self.dirs_emotion_actors:
            # print(emotion_actor)
            for root, dir_frames, _ in os.walk(emotion_actor):
                dir_frames = outclude_hidden_dirs(dir_frames)
                dir_frames.sort()
                # print(os.path.join(root, dir_frames[0]))
                for dir_frame in dir_frames:
                    for r, dir_emotion, emotion_file in os.walk(os.path.join(root, dir_frame)):
                        emotion_file = outclude_hidden_files(emotion_file)
                        if emotion_file:
                            r_split = r.split('/')
                            key_ = r_split[-2] + '/' + r_split[-1]
                            self.key_sequence_with_labels.append(key_)
                            emotion_files.append(os.path.join(r, emotion_file[0]))
                            f = open(emotion_files[-1], 'r+')
                            line = f.readline()  # only one row
                            # print(int(line.split('.')[0]))
                            label = int(line.split('.')[0])
                            lables.append(label)
                            self.dict_key_seq_labels[key_] = label
                break

    def get_dict_lab_ikeys(self):
        for emotion_lab in emotion_labs:
            self.dict_lab_ikeys[emotion_lab] = []

        for k, v in self.dict_key_seq_labels.items():
            self.dict_lab_ikeys[v].append(k)

    def get_dict_lab_images(self):
        indexes = []
        for i in np.arange(self.num_frames):
            indexes.append(-i - 1)

        for emotion_lab in emotion_labs:
            self.dict_lab_images[emotion_lab] = []
            for ikey in self.dict_lab_ikeys[emotion_lab]:
                for r, _, image_files in os.walk(os.path.join(self.r_dir_images, ikey)):
                    image_files = outclude_hidden_files(image_files)
                    image_files.sort()
                    self.dict_lab_images[emotion_lab].extend(np.array(image_files)[indexes])
                    self.dict_lab_images[Neutral].append(image_files[0])

    def get_dict_image_lab(self):
        for lab, images in self.dict_lab_images.items():
            for image_name in images:
                self.dict_image_lab[image_name] = lab

    def get_full_path(self, image_name):
        list_str = image_name.split('_')
        image_path = self.r_dir_images
        for i in np.arange(2):
            image_path = os.path.join(image_path, list_str[i])
            image_path = image_path + '/'
            if i == 1:
                break
        return image_path + image_name

    def get_dict_image_landmarks(self):
        for k, v in self.dict_image_lab.items():
            r_splits = k.split('_')
            dir_landmarks = self.r_dir_landmarks + r_splits[0] + '/' + r_splits[1] + '/'
            r_splits_2 = k.split('.')
            landmark_file = r_splits_2[0] + '_' + 'landmarks.txt'
            # print(landmark_file)
            f = open(os.path.join(dir_landmarks, landmark_file), 'r+')
            line = f.readline()  # only one row
            list_lines = []
            landmarks_list = []
            while line:
                list_lines.append(line)
                xy = line.split()
                x = float(xy[0])
                y = float(xy[1])
                landmarks_list.append([x, y])
                line = f.readline()
            # print(landmarks_list)
            landmarks_array = np.array(landmarks_list)
            # print(landmarks_array)
            self.dict_image_landmarks[k] = landmarks_array

    def get_dict_image_imgarray(self):
        for image_name, lab in self.dict_image_lab.items():
            image_name_full_path = self.get_full_path(image_name)
            img = cv2.imread(image_name_full_path, 0)
            self.dict_image_imgarray[image_name] = img
