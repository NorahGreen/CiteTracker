import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str

############
# current 00000492.png of test_015_Sord_video_Q01_done is damaged and replaced by a copy of 00000491.png
############


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()
        # self.sequence_list = ['BianLian_video_03_done', 'Bullet_video_08_done', 'Cartoon_Robot_video_Z01_done', 'CrashCar_video_04', 'CartoonHuLuWa_video_04-Done']

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        # target_class = class_name
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), object_class=text_dsp)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        cover_list = []
        for seq in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, seq)):
                # for attribute
                # attributes_path = os.path.join(os.path.join(self.base_path, seq), 'attributes.txt')
                # attributes_file = np.loadtxt(attributes_path, delimiter=' ', dtype=np.float32)
                # # if (attributes_file[0]==1 or attributes_file[1]==1 or attributes_file[2]==1 or attributes_file[4]==1
                # # or attributes_file[7]==1 or attributes_file[8]==1 or attributes_file[10]==1 or attributes_file[11]==1):
                # if (attributes_file[16] == 1 ):

                # for tnl2k unknown cases
                text_dsp_path = '{}/{}/language.txt'.format(self.base_path, seq)
                text_dsp = load_str(text_dsp_path)
                sequence_list.append(seq)
                # for i in label:
                #     if i in text_dsp or i in seq.lower():
                #         cover_list.append(seq)
                #         # print(seq)
                #         break

                # sequence_list.append(seq)
        # sequence_list = set(sequence_list) - set(cover_list)
        # sequence_list = cover_list
        return sequence_list


label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush', 'nba', 'people', 'baseball', 'ball', 'tennis', 'phone', 'table']