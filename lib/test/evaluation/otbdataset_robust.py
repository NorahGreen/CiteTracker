import numpy as np
from CiteTracker.lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from CiteTracker.lib.test.utils.load_text import load_text


class OTBDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        # anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])
        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        ground_truth_rect = ground_truth_rect[start_frame - 1:end_frame -1, :]

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'Basketball-0', 'path': 'Basketball/img', 'startFrame': 1, 'endFrame': 725, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-1', 'path': 'Basketball/img', 'startFrame': 15, 'endFrame': 725, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-2', 'path': 'Basketball/img', 'startFrame': 109, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-3', 'path': 'Basketball/img', 'startFrame': 159, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-4', 'path': 'Basketball/img', 'startFrame': 238, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-5', 'path': 'Basketball/img', 'startFrame': 303, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-6', 'path': 'Basketball/img', 'startFrame': 404, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-7', 'path': 'Basketball/img', 'startFrame': 476, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-8', 'path': 'Basketball/img', 'startFrame': 533, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Basketball-9', 'path': 'Basketball/img', 'startFrame': 613, 'endFrame': 725, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Biker-0', 'path': 'Biker/img', 'startFrame': 1, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-1', 'path': 'Biker/img', 'startFrame': 6, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-2', 'path': 'Biker/img', 'startFrame': 17, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-3', 'path': 'Biker/img', 'startFrame': 41, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-4', 'path': 'Biker/img', 'startFrame': 55, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-5', 'path': 'Biker/img', 'startFrame': 69, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-6', 'path': 'Biker/img', 'startFrame': 73, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-7', 'path': 'Biker/img', 'startFrame': 96, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-8', 'path': 'Biker/img', 'startFrame': 106, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Biker-9', 'path': 'Biker/img', 'startFrame': 114, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Bird1-0', 'path': 'Bird1/img', 'startFrame': 1, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-1', 'path': 'Bird1/img', 'startFrame': 9, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-2', 'path': 'Bird1/img', 'startFrame': 81, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-3', 'path': 'Bird1/img', 'startFrame': 85, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-4', 'path': 'Bird1/img', 'startFrame': 157, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-5', 'path': 'Bird1/img', 'startFrame': 185, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-6', 'path': 'Bird1/img', 'startFrame': 225, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-7', 'path': 'Bird1/img', 'startFrame': 269, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-8', 'path': 'Bird1/img', 'startFrame': 297, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird1-9', 'path': 'Bird1/img', 'startFrame': 337, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-0', 'path': 'Bird2/img', 'startFrame': 1, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-1', 'path': 'Bird2/img', 'startFrame': 4, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-2', 'path': 'Bird2/img', 'startFrame': 17, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-3', 'path': 'Bird2/img', 'startFrame': 27, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-4', 'path': 'Bird2/img', 'startFrame': 35, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-5', 'path': 'Bird2/img', 'startFrame': 45, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-6', 'path': 'Bird2/img', 'startFrame': 52, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-7', 'path': 'Bird2/img', 'startFrame': 56, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-8', 'path': 'Bird2/img', 'startFrame': 70, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'Bird2-9', 'path': 'Bird2/img', 'startFrame': 82, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
            ,
            {'name': 'BlurBody-0', 'path': 'BlurBody/img', 'startFrame': 1, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-1', 'path': 'BlurBody/img', 'startFrame': 7, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-2', 'path': 'BlurBody/img', 'startFrame': 40, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-3', 'path': 'BlurBody/img', 'startFrame': 76, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-4', 'path': 'BlurBody/img', 'startFrame': 119, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-5', 'path': 'BlurBody/img', 'startFrame': 162, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-6', 'path': 'BlurBody/img', 'startFrame': 175, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-7', 'path': 'BlurBody/img', 'startFrame': 212, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-8', 'path': 'BlurBody/img', 'startFrame': 245, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurBody-9', 'path': 'BlurBody/img', 'startFrame': 271, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'BlurCar1-0', 'path': 'BlurCar1/img', 'startFrame': 247, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-1', 'path': 'BlurCar1/img', 'startFrame': 261, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-2', 'path': 'BlurCar1/img', 'startFrame': 305, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-3', 'path': 'BlurCar1/img', 'startFrame': 359, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-4', 'path': 'BlurCar1/img', 'startFrame': 423, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-5', 'path': 'BlurCar1/img', 'startFrame': 462, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-6', 'path': 'BlurCar1/img', 'startFrame': 516, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-7', 'path': 'BlurCar1/img', 'startFrame': 565, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-8', 'path': 'BlurCar1/img', 'startFrame': 629, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar1-9', 'path': 'BlurCar1/img', 'startFrame': 678, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-0', 'path': 'BlurCar2/img', 'startFrame': 1, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-1', 'path': 'BlurCar2/img', 'startFrame': 47, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-2', 'path': 'BlurCar2/img', 'startFrame': 99, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-3', 'path': 'BlurCar2/img', 'startFrame': 163, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-4', 'path': 'BlurCar2/img', 'startFrame': 233, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-5', 'path': 'BlurCar2/img', 'startFrame': 273, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-6', 'path': 'BlurCar2/img', 'startFrame': 302, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-7', 'path': 'BlurCar2/img', 'startFrame': 401, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-8', 'path': 'BlurCar2/img', 'startFrame': 418, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar2-9', 'path': 'BlurCar2/img', 'startFrame': 488, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-0', 'path': 'BlurCar3/img', 'startFrame': 3, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-1', 'path': 'BlurCar3/img', 'startFrame': 38, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-2', 'path': 'BlurCar3/img', 'startFrame': 55, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-3', 'path': 'BlurCar3/img', 'startFrame': 90, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-4', 'path': 'BlurCar3/img', 'startFrame': 122, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-5', 'path': 'BlurCar3/img', 'startFrame': 160, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-6', 'path': 'BlurCar3/img', 'startFrame': 213, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-7', 'path': 'BlurCar3/img', 'startFrame': 244, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-8', 'path': 'BlurCar3/img', 'startFrame': 251, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar3-9', 'path': 'BlurCar3/img', 'startFrame': 300, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-0', 'path': 'BlurCar4/img', 'startFrame': 18, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-1', 'path': 'BlurCar4/img', 'startFrame': 39, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-2', 'path': 'BlurCar4/img', 'startFrame': 72, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-3', 'path': 'BlurCar4/img', 'startFrame': 97, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-4', 'path': 'BlurCar4/img', 'startFrame': 144, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-5', 'path': 'BlurCar4/img', 'startFrame': 183, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-6', 'path': 'BlurCar4/img', 'startFrame': 226, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-7', 'path': 'BlurCar4/img', 'startFrame': 259, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-8', 'path': 'BlurCar4/img', 'startFrame': 273, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurCar4-9', 'path': 'BlurCar4/img', 'startFrame': 309, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'BlurFace-0', 'path': 'BlurFace/img', 'startFrame': 1, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-1', 'path': 'BlurFace/img', 'startFrame': 45, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-2', 'path': 'BlurFace/img', 'startFrame': 94, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-3', 'path': 'BlurFace/img', 'startFrame': 113, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-4', 'path': 'BlurFace/img', 'startFrame': 152, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-5', 'path': 'BlurFace/img', 'startFrame': 206, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-6', 'path': 'BlurFace/img', 'startFrame': 295, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-7', 'path': 'BlurFace/img', 'startFrame': 309, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-8', 'path': 'BlurFace/img', 'startFrame': 363, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurFace-9', 'path': 'BlurFace/img', 'startFrame': 397, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'BlurOwl-0', 'path': 'BlurOwl/img', 'startFrame': 1, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-1', 'path': 'BlurOwl/img', 'startFrame': 45, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-2', 'path': 'BlurOwl/img', 'startFrame': 70, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-3', 'path': 'BlurOwl/img', 'startFrame': 177, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-4', 'path': 'BlurOwl/img', 'startFrame': 227, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-5', 'path': 'BlurOwl/img', 'startFrame': 284, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-6', 'path': 'BlurOwl/img', 'startFrame': 372, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-7', 'path': 'BlurOwl/img', 'startFrame': 391, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-8', 'path': 'BlurOwl/img', 'startFrame': 492, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'BlurOwl-9', 'path': 'BlurOwl/img', 'startFrame': 555, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-0', 'path': 'Board/img', 'startFrame': 1, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-1', 'path': 'Board/img', 'startFrame': 49, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-2', 'path': 'Board/img', 'startFrame': 76, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-3', 'path': 'Board/img', 'startFrame': 159, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-4', 'path': 'Board/img', 'startFrame': 235, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-5', 'path': 'Board/img', 'startFrame': 290, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-6', 'path': 'Board/img', 'startFrame': 352, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-7', 'path': 'Board/img', 'startFrame': 428, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-8', 'path': 'Board/img', 'startFrame': 497, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Board-9', 'path': 'Board/img', 'startFrame': 608, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
             'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Bolt-0', 'path': 'Bolt/img', 'startFrame': 1, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-1', 'path': 'Bolt/img', 'startFrame': 15, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-2', 'path': 'Bolt/img', 'startFrame': 39, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-3', 'path': 'Bolt/img', 'startFrame': 78, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-4', 'path': 'Bolt/img', 'startFrame': 130, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-5', 'path': 'Bolt/img', 'startFrame': 158, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-6', 'path': 'Bolt/img', 'startFrame': 197, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-7', 'path': 'Bolt/img', 'startFrame': 242, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-8', 'path': 'Bolt/img', 'startFrame': 267, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt-9', 'path': 'Bolt/img', 'startFrame': 284, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-0', 'path': 'Bolt2/img', 'startFrame': 1, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-1', 'path': 'Bolt2/img', 'startFrame': 18, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-2', 'path': 'Bolt2/img', 'startFrame': 59, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-3', 'path': 'Bolt2/img', 'startFrame': 64, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-4', 'path': 'Bolt2/img', 'startFrame': 114, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-5', 'path': 'Bolt2/img', 'startFrame': 137, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-6', 'path': 'Bolt2/img', 'startFrame': 148, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-7', 'path': 'Bolt2/img', 'startFrame': 186, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-8', 'path': 'Bolt2/img', 'startFrame': 212, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Bolt2-9', 'path': 'Bolt2/img', 'startFrame': 250, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Box-0', 'path': 'Box/img', 'startFrame': 1, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-1', 'path': 'Box/img', 'startFrame': 82, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-2', 'path': 'Box/img', 'startFrame': 128, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-3', 'path': 'Box/img', 'startFrame': 302, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-4', 'path': 'Box/img', 'startFrame': 465, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-5', 'path': 'Box/img', 'startFrame': 546, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-6', 'path': 'Box/img', 'startFrame': 697, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-7', 'path': 'Box/img', 'startFrame': 778, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-8', 'path': 'Box/img', 'startFrame': 836, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Box-9', 'path': 'Box/img', 'startFrame': 1021, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Boy-0', 'path': 'Boy/img', 'startFrame': 1, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-1', 'path': 'Boy/img', 'startFrame': 55, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-2', 'path': 'Boy/img', 'startFrame': 91, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-3', 'path': 'Boy/img', 'startFrame': 133, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-4', 'path': 'Boy/img', 'startFrame': 205, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-5', 'path': 'Boy/img', 'startFrame': 289, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-6', 'path': 'Boy/img', 'startFrame': 307, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-7', 'path': 'Boy/img', 'startFrame': 397, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-8', 'path': 'Boy/img', 'startFrame': 451, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Boy-9', 'path': 'Boy/img', 'startFrame': 492, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Car1-0', 'path': 'Car1/img', 'startFrame': 1, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-1', 'path': 'Car1/img', 'startFrame': 41, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-2', 'path': 'Car1/img', 'startFrame': 174, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-3', 'path': 'Car1/img', 'startFrame': 307, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-4', 'path': 'Car1/img', 'startFrame': 398, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-5', 'path': 'Car1/img', 'startFrame': 500, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-6', 'path': 'Car1/img', 'startFrame': 551, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-7', 'path': 'Car1/img', 'startFrame': 694, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-8', 'path': 'Car1/img', 'startFrame': 745, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car1-9', 'path': 'Car1/img', 'startFrame': 837, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-0', 'path': 'Car2/img', 'startFrame': 1, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-1', 'path': 'Car2/img', 'startFrame': 37, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-2', 'path': 'Car2/img', 'startFrame': 155, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-3', 'path': 'Car2/img', 'startFrame': 237, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-4', 'path': 'Car2/img', 'startFrame': 328, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-5', 'path': 'Car2/img', 'startFrame': 419, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-6', 'path': 'Car2/img', 'startFrame': 519, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-7', 'path': 'Car2/img', 'startFrame': 610, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-8', 'path': 'Car2/img', 'startFrame': 674, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car2-9', 'path': 'Car2/img', 'startFrame': 765, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-0', 'path': 'Car24/img', 'startFrame': 1, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-1', 'path': 'Car24/img', 'startFrame': 123, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-2', 'path': 'Car24/img', 'startFrame': 397, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-3', 'path': 'Car24/img', 'startFrame': 855, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-4', 'path': 'Car24/img', 'startFrame': 977, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-5', 'path': 'Car24/img', 'startFrame': 1251, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-6', 'path': 'Car24/img', 'startFrame': 1831, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-7', 'path': 'Car24/img', 'startFrame': 1861, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-8', 'path': 'Car24/img', 'startFrame': 2288, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car24-9', 'path': 'Car24/img', 'startFrame': 2685, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-0', 'path': 'Car4/img', 'startFrame': 1, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-1', 'path': 'Car4/img', 'startFrame': 7, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-2', 'path': 'Car4/img', 'startFrame': 92, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-3', 'path': 'Car4/img', 'startFrame': 137, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-4', 'path': 'Car4/img', 'startFrame': 248, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-5', 'path': 'Car4/img', 'startFrame': 326, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-6', 'path': 'Car4/img', 'startFrame': 332, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-7', 'path': 'Car4/img', 'startFrame': 436, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-8', 'path': 'Car4/img', 'startFrame': 462, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Car4-9', 'path': 'Car4/img', 'startFrame': 566, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-0', 'path': 'CarDark/img', 'startFrame': 1, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-1', 'path': 'CarDark/img', 'startFrame': 32, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-2', 'path': 'CarDark/img', 'startFrame': 75, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-3', 'path': 'CarDark/img', 'startFrame': 86, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-4', 'path': 'CarDark/img', 'startFrame': 125, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-5', 'path': 'CarDark/img', 'startFrame': 172, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-6', 'path': 'CarDark/img', 'startFrame': 223, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-7', 'path': 'CarDark/img', 'startFrame': 246, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-8', 'path': 'CarDark/img', 'startFrame': 301, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarDark-9', 'path': 'CarDark/img', 'startFrame': 320, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-0', 'path': 'CarScale/img', 'startFrame': 1, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-1', 'path': 'CarScale/img', 'startFrame': 18, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-2', 'path': 'CarScale/img', 'startFrame': 41, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-3', 'path': 'CarScale/img', 'startFrame': 53, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-4', 'path': 'CarScale/img', 'startFrame': 93, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-5', 'path': 'CarScale/img', 'startFrame': 111, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-6', 'path': 'CarScale/img', 'startFrame': 138, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-7', 'path': 'CarScale/img', 'startFrame': 163, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-8', 'path': 'CarScale/img', 'startFrame': 188, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'CarScale-9', 'path': 'CarScale/img', 'startFrame': 216, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'ClifBar-0', 'path': 'ClifBar/img', 'startFrame': 1, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-1', 'path': 'ClifBar/img', 'startFrame': 15, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-2', 'path': 'ClifBar/img', 'startFrame': 76, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-3', 'path': 'ClifBar/img', 'startFrame': 113, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-4', 'path': 'ClifBar/img', 'startFrame': 179, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-5', 'path': 'ClifBar/img', 'startFrame': 203, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-6', 'path': 'ClifBar/img', 'startFrame': 250, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-7', 'path': 'ClifBar/img', 'startFrame': 292, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-8', 'path': 'ClifBar/img', 'startFrame': 372, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'ClifBar-9', 'path': 'ClifBar/img', 'startFrame': 395, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-0', 'path': 'Coke/img', 'startFrame': 1, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-1', 'path': 'Coke/img', 'startFrame': 21, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-2', 'path': 'Coke/img', 'startFrame': 38, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-3', 'path': 'Coke/img', 'startFrame': 79, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-4', 'path': 'Coke/img', 'startFrame': 114, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-5', 'path': 'Coke/img', 'startFrame': 140, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-6', 'path': 'Coke/img', 'startFrame': 166, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-7', 'path': 'Coke/img', 'startFrame': 198, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-8', 'path': 'Coke/img', 'startFrame': 227, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coke-9', 'path': 'Coke/img', 'startFrame': 244, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Couple-0', 'path': 'Couple/img', 'startFrame': 1, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-1', 'path': 'Couple/img', 'startFrame': 2, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-2', 'path': 'Couple/img', 'startFrame': 27, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-3', 'path': 'Couple/img', 'startFrame': 40, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-4', 'path': 'Couple/img', 'startFrame': 51, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-5', 'path': 'Couple/img', 'startFrame': 69, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-6', 'path': 'Couple/img', 'startFrame': 83, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-7', 'path': 'Couple/img', 'startFrame': 86, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-8', 'path': 'Couple/img', 'startFrame': 108, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Couple-9', 'path': 'Couple/img', 'startFrame': 115, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Coupon-0', 'path': 'Coupon/img', 'startFrame': 1, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-1', 'path': 'Coupon/img', 'startFrame': 7, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-2', 'path': 'Coupon/img', 'startFrame': 55, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-3', 'path': 'Coupon/img', 'startFrame': 68, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-4', 'path': 'Coupon/img', 'startFrame': 122, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-5', 'path': 'Coupon/img', 'startFrame': 157, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-6', 'path': 'Coupon/img', 'startFrame': 183, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-7', 'path': 'Coupon/img', 'startFrame': 209, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-8', 'path': 'Coupon/img', 'startFrame': 241, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Coupon-9', 'path': 'Coupon/img', 'startFrame': 282, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Crossing-0', 'path': 'Crossing/img', 'startFrame': 1, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-1', 'path': 'Crossing/img', 'startFrame': 11, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-2', 'path': 'Crossing/img', 'startFrame': 25, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-3', 'path': 'Crossing/img', 'startFrame': 27, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-4', 'path': 'Crossing/img', 'startFrame': 45, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-5', 'path': 'Crossing/img', 'startFrame': 61, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-6', 'path': 'Crossing/img', 'startFrame': 69, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-7', 'path': 'Crossing/img', 'startFrame': 75, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-8', 'path': 'Crossing/img', 'startFrame': 92, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crossing-9', 'path': 'Crossing/img', 'startFrame': 105, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-0', 'path': 'Crowds/img', 'startFrame': 1, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-1', 'path': 'Crowds/img', 'startFrame': 24, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-2', 'path': 'Crowds/img', 'startFrame': 41, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-3', 'path': 'Crowds/img', 'startFrame': 103, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-4', 'path': 'Crowds/img', 'startFrame': 116, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-5', 'path': 'Crowds/img', 'startFrame': 160, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-6', 'path': 'Crowds/img', 'startFrame': 205, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-7', 'path': 'Crowds/img', 'startFrame': 222, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-8', 'path': 'Crowds/img', 'startFrame': 273, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Crowds-9', 'path': 'Crowds/img', 'startFrame': 303, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-0', 'path': 'Dancer/img', 'startFrame': 1, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-1', 'path': 'Dancer/img', 'startFrame': 20, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-2', 'path': 'Dancer/img', 'startFrame': 40, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-3', 'path': 'Dancer/img', 'startFrame': 53, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-4', 'path': 'Dancer/img', 'startFrame': 75, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-5', 'path': 'Dancer/img', 'startFrame': 93, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-6', 'path': 'Dancer/img', 'startFrame': 133, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-7', 'path': 'Dancer/img', 'startFrame': 155, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-8', 'path': 'Dancer/img', 'startFrame': 166, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer-9', 'path': 'Dancer/img', 'startFrame': 196, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-0', 'path': 'Dancer2/img', 'startFrame': 1, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-1', 'path': 'Dancer2/img', 'startFrame': 5, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-2', 'path': 'Dancer2/img', 'startFrame': 17, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-3', 'path': 'Dancer2/img', 'startFrame': 38, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-4', 'path': 'Dancer2/img', 'startFrame': 52, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-5', 'path': 'Dancer2/img', 'startFrame': 68, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-6', 'path': 'Dancer2/img', 'startFrame': 85, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-7', 'path': 'Dancer2/img', 'startFrame': 94, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-8', 'path': 'Dancer2/img', 'startFrame': 109, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dancer2-9', 'path': 'Dancer2/img', 'startFrame': 130, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David-0', 'path': 'David/img', 'startFrame': 300, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-1', 'path': 'David/img', 'startFrame': 301, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-2', 'path': 'David/img', 'startFrame': 327, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-3', 'path': 'David/img', 'startFrame': 349, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-4', 'path': 'David/img', 'startFrame': 352, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-5', 'path': 'David/img', 'startFrame': 381, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-6', 'path': 'David/img', 'startFrame': 402, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-7', 'path': 'David/img', 'startFrame': 413, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-8', 'path': 'David/img', 'startFrame': 422, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David-9', 'path': 'David/img', 'startFrame': 451, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-0', 'path': 'David2/img', 'startFrame': 1, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-1', 'path': 'David2/img', 'startFrame': 6, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-2', 'path': 'David2/img', 'startFrame': 85, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-3', 'path': 'David2/img', 'startFrame': 144, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-4', 'path': 'David2/img', 'startFrame': 181, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-5', 'path': 'David2/img', 'startFrame': 218, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-6', 'path': 'David2/img', 'startFrame': 281, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-7', 'path': 'David2/img', 'startFrame': 334, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-8', 'path': 'David2/img', 'startFrame': 387, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David2-9', 'path': 'David2/img', 'startFrame': 478, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'David3-0', 'path': 'David3/img', 'startFrame': 1, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-1', 'path': 'David3/img', 'startFrame': 26, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-2', 'path': 'David3/img', 'startFrame': 31, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-3', 'path': 'David3/img', 'startFrame': 58, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-4', 'path': 'David3/img', 'startFrame': 88, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-5', 'path': 'David3/img', 'startFrame': 106, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-6', 'path': 'David3/img', 'startFrame': 128, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-7', 'path': 'David3/img', 'startFrame': 176, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-8', 'path': 'David3/img', 'startFrame': 181, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'David3-9', 'path': 'David3/img', 'startFrame': 211, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Deer-0', 'path': 'Deer/img', 'startFrame': 1, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-1', 'path': 'Deer/img', 'startFrame': 7, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-2', 'path': 'Deer/img', 'startFrame': 14, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-3', 'path': 'Deer/img', 'startFrame': 16, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-4', 'path': 'Deer/img', 'startFrame': 25, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-5', 'path': 'Deer/img', 'startFrame': 36, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-6', 'path': 'Deer/img', 'startFrame': 42, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-7', 'path': 'Deer/img', 'startFrame': 45, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-8', 'path': 'Deer/img', 'startFrame': 51, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Deer-9', 'path': 'Deer/img', 'startFrame': 61, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Diving-0', 'path': 'Diving/img', 'startFrame': 1, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-1', 'path': 'Diving/img', 'startFrame': 5, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-2', 'path': 'Diving/img', 'startFrame': 30, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-3', 'path': 'Diving/img', 'startFrame': 59, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-4', 'path': 'Diving/img', 'startFrame': 72, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-5', 'path': 'Diving/img', 'startFrame': 95, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-6', 'path': 'Diving/img', 'startFrame': 112, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-7', 'path': 'Diving/img', 'startFrame': 139, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-8', 'path': 'Diving/img', 'startFrame': 154, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Diving-9', 'path': 'Diving/img', 'startFrame': 181, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Dog-0', 'path': 'Dog/img', 'startFrame': 1, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-1', 'path': 'Dog/img', 'startFrame': 2, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-2', 'path': 'Dog/img', 'startFrame': 14, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-3', 'path': 'Dog/img', 'startFrame': 28, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-4', 'path': 'Dog/img', 'startFrame': 44, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-5', 'path': 'Dog/img', 'startFrame': 56, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-6', 'path': 'Dog/img', 'startFrame': 67, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-7', 'path': 'Dog/img', 'startFrame': 74, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-8', 'path': 'Dog/img', 'startFrame': 92, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog-9', 'path': 'Dog/img', 'startFrame': 98, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-0', 'path': 'Dog1/img', 'startFrame': 1, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-1', 'path': 'Dog1/img', 'startFrame': 109, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-2', 'path': 'Dog1/img', 'startFrame': 271, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-3', 'path': 'Dog1/img', 'startFrame': 379, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-4', 'path': 'Dog1/img', 'startFrame': 541, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-5', 'path': 'Dog1/img', 'startFrame': 662, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-6', 'path': 'Dog1/img', 'startFrame': 757, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-7', 'path': 'Dog1/img', 'startFrame': 892, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-8', 'path': 'Dog1/img', 'startFrame': 1081, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Dog1-9', 'path': 'Dog1/img', 'startFrame': 1216, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
            ,
            {'name': 'Doll-0', 'path': 'Doll/img', 'startFrame': 1, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-1', 'path': 'Doll/img', 'startFrame': 117, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-2', 'path': 'Doll/img', 'startFrame': 697, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-3', 'path': 'Doll/img', 'startFrame': 929, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-4', 'path': 'Doll/img', 'startFrame': 1239, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-5', 'path': 'Doll/img', 'startFrame': 1819, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-6', 'path': 'Doll/img', 'startFrame': 2245, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-7', 'path': 'Doll/img', 'startFrame': 2632, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-8', 'path': 'Doll/img', 'startFrame': 2748, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Doll-9', 'path': 'Doll/img', 'startFrame': 3251, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'DragonBaby-0', 'path': 'DragonBaby/img', 'startFrame': 1, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-1', 'path': 'DragonBaby/img', 'startFrame': 3, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-2', 'path': 'DragonBaby/img', 'startFrame': 19, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-3', 'path': 'DragonBaby/img', 'startFrame': 30, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-4', 'path': 'DragonBaby/img', 'startFrame': 39, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-5', 'path': 'DragonBaby/img', 'startFrame': 52, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-6', 'path': 'DragonBaby/img', 'startFrame': 60, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-7', 'path': 'DragonBaby/img', 'startFrame': 74, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-8', 'path': 'DragonBaby/img', 'startFrame': 81, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'DragonBaby-9', 'path': 'DragonBaby/img', 'startFrame': 96, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-0', 'path': 'Dudek/img', 'startFrame': 1, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-1', 'path': 'Dudek/img', 'startFrame': 35, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-2', 'path': 'Dudek/img', 'startFrame': 137, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-3', 'path': 'Dudek/img', 'startFrame': 274, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-4', 'path': 'Dudek/img', 'startFrame': 388, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-5', 'path': 'Dudek/img', 'startFrame': 536, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-6', 'path': 'Dudek/img', 'startFrame': 662, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-7', 'path': 'Dudek/img', 'startFrame': 753, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-8', 'path': 'Dudek/img', 'startFrame': 901, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Dudek-9', 'path': 'Dudek/img', 'startFrame': 947, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-0', 'path': 'FaceOcc1/img', 'startFrame': 1, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-1', 'path': 'FaceOcc1/img', 'startFrame': 36, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-2', 'path': 'FaceOcc1/img', 'startFrame': 143, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-3', 'path': 'FaceOcc1/img', 'startFrame': 259, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-4', 'path': 'FaceOcc1/img', 'startFrame': 330, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-5', 'path': 'FaceOcc1/img', 'startFrame': 383, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-6', 'path': 'FaceOcc1/img', 'startFrame': 490, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-7', 'path': 'FaceOcc1/img', 'startFrame': 579, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-8', 'path': 'FaceOcc1/img', 'startFrame': 713, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc1-9', 'path': 'FaceOcc1/img', 'startFrame': 730, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-0', 'path': 'FaceOcc2/img', 'startFrame': 1, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-1', 'path': 'FaceOcc2/img', 'startFrame': 57, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-2', 'path': 'FaceOcc2/img', 'startFrame': 114, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-3', 'path': 'FaceOcc2/img', 'startFrame': 219, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-4', 'path': 'FaceOcc2/img', 'startFrame': 300, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-5', 'path': 'FaceOcc2/img', 'startFrame': 373, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-6', 'path': 'FaceOcc2/img', 'startFrame': 446, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-7', 'path': 'FaceOcc2/img', 'startFrame': 535, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-8', 'path': 'FaceOcc2/img', 'startFrame': 584, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FaceOcc2-9', 'path': 'FaceOcc2/img', 'startFrame': 657, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Fish-0', 'path': 'Fish/img', 'startFrame': 1, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-1', 'path': 'Fish/img', 'startFrame': 10, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-2', 'path': 'Fish/img', 'startFrame': 76, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-3', 'path': 'Fish/img', 'startFrame': 132, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-4', 'path': 'Fish/img', 'startFrame': 160, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-5', 'path': 'Fish/img', 'startFrame': 198, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-6', 'path': 'Fish/img', 'startFrame': 273, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-7', 'path': 'Fish/img', 'startFrame': 306, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-8', 'path': 'Fish/img', 'startFrame': 344, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Fish-9', 'path': 'Fish/img', 'startFrame': 414, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'FleetFace-0', 'path': 'FleetFace/img', 'startFrame': 1, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-1', 'path': 'FleetFace/img', 'startFrame': 43, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-2', 'path': 'FleetFace/img', 'startFrame': 85, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-3', 'path': 'FleetFace/img', 'startFrame': 204, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-4', 'path': 'FleetFace/img', 'startFrame': 232, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-5', 'path': 'FleetFace/img', 'startFrame': 330, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-6', 'path': 'FleetFace/img', 'startFrame': 379, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-7', 'path': 'FleetFace/img', 'startFrame': 463, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-8', 'path': 'FleetFace/img', 'startFrame': 505, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'FleetFace-9', 'path': 'FleetFace/img', 'startFrame': 596, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football-0', 'path': 'Football/img', 'startFrame': 1, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-1', 'path': 'Football/img', 'startFrame': 15, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-2', 'path': 'Football/img', 'startFrame': 44, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-3', 'path': 'Football/img', 'startFrame': 98, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-4', 'path': 'Football/img', 'startFrame': 137, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-5', 'path': 'Football/img', 'startFrame': 155, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-6', 'path': 'Football/img', 'startFrame': 184, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-7', 'path': 'Football/img', 'startFrame': 220, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-8', 'path': 'Football/img', 'startFrame': 263, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football-9', 'path': 'Football/img', 'startFrame': 292, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Football1-0', 'path': 'Football1/img', 'startFrame': 1, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-1', 'path': 'Football1/img', 'startFrame': 3, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-2', 'path': 'Football1/img', 'startFrame': 15, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-3', 'path': 'Football1/img', 'startFrame': 18, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-4', 'path': 'Football1/img', 'startFrame': 26, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-5', 'path': 'Football1/img', 'startFrame': 31, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-6', 'path': 'Football1/img', 'startFrame': 42, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-7', 'path': 'Football1/img', 'startFrame': 44, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-8', 'path': 'Football1/img', 'startFrame': 51, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Football1-9', 'path': 'Football1/img', 'startFrame': 59, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-0', 'path': 'Freeman1/img', 'startFrame': 1, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-1', 'path': 'Freeman1/img', 'startFrame': 7, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-2', 'path': 'Freeman1/img', 'startFrame': 45, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-3', 'path': 'Freeman1/img', 'startFrame': 90, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-4', 'path': 'Freeman1/img', 'startFrame': 129, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-5', 'path': 'Freeman1/img', 'startFrame': 154, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-6', 'path': 'Freeman1/img', 'startFrame': 173, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-7', 'path': 'Freeman1/img', 'startFrame': 199, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-8', 'path': 'Freeman1/img', 'startFrame': 250, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman1-9', 'path': 'Freeman1/img', 'startFrame': 285, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-0', 'path': 'Freeman3/img', 'startFrame': 1, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-1', 'path': 'Freeman3/img', 'startFrame': 19, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-2', 'path': 'Freeman3/img', 'startFrame': 74, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-3', 'path': 'Freeman3/img', 'startFrame': 97, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-4', 'path': 'Freeman3/img', 'startFrame': 152, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-5', 'path': 'Freeman3/img', 'startFrame': 194, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-6', 'path': 'Freeman3/img', 'startFrame': 244, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-7', 'path': 'Freeman3/img', 'startFrame': 300, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-8', 'path': 'Freeman3/img', 'startFrame': 355, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman3-9', 'path': 'Freeman3/img', 'startFrame': 382, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-0', 'path': 'Freeman4/img', 'startFrame': 1, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-1', 'path': 'Freeman4/img', 'startFrame': 26, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-2', 'path': 'Freeman4/img', 'startFrame': 57, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-3', 'path': 'Freeman4/img', 'startFrame': 85, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-4', 'path': 'Freeman4/img', 'startFrame': 104, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-5', 'path': 'Freeman4/img', 'startFrame': 138, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-6', 'path': 'Freeman4/img', 'startFrame': 155, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-7', 'path': 'Freeman4/img', 'startFrame': 185, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-8', 'path': 'Freeman4/img', 'startFrame': 222, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Freeman4-9', 'path': 'Freeman4/img', 'startFrame': 253, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-0', 'path': 'Girl/img', 'startFrame': 1, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-1', 'path': 'Girl/img', 'startFrame': 16, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-2', 'path': 'Girl/img', 'startFrame': 56, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-3', 'path': 'Girl/img', 'startFrame': 126, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-4', 'path': 'Girl/img', 'startFrame': 181, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-5', 'path': 'Girl/img', 'startFrame': 216, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-6', 'path': 'Girl/img', 'startFrame': 286, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-7', 'path': 'Girl/img', 'startFrame': 341, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-8', 'path': 'Girl/img', 'startFrame': 396, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl-9', 'path': 'Girl/img', 'startFrame': 416, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Girl2-0', 'path': 'Girl2/img', 'startFrame': 1, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-1', 'path': 'Girl2/img', 'startFrame': 76, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-2', 'path': 'Girl2/img', 'startFrame': 226, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-3', 'path': 'Girl2/img', 'startFrame': 331, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-4', 'path': 'Girl2/img', 'startFrame': 496, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-5', 'path': 'Girl2/img', 'startFrame': 631, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-6', 'path': 'Girl2/img', 'startFrame': 781, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-7', 'path': 'Girl2/img', 'startFrame': 961, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-8', 'path': 'Girl2/img', 'startFrame': 1156, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Girl2-9', 'path': 'Girl2/img', 'startFrame': 1291, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-0', 'path': 'Gym/img', 'startFrame': 1, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-1', 'path': 'Gym/img', 'startFrame': 54, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-2', 'path': 'Gym/img', 'startFrame': 99, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-3', 'path': 'Gym/img', 'startFrame': 229, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-4', 'path': 'Gym/img', 'startFrame': 305, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-5', 'path': 'Gym/img', 'startFrame': 312, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-6', 'path': 'Gym/img', 'startFrame': 388, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-7', 'path': 'Gym/img', 'startFrame': 510, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-8', 'path': 'Gym/img', 'startFrame': 586, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Gym-9', 'path': 'Gym/img', 'startFrame': 654, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-0', 'path': 'Human2/img', 'startFrame': 1, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-1', 'path': 'Human2/img', 'startFrame': 45, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-2', 'path': 'Human2/img', 'startFrame': 191, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-3', 'path': 'Human2/img', 'startFrame': 269, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-4', 'path': 'Human2/img', 'startFrame': 381, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-5', 'path': 'Human2/img', 'startFrame': 482, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-6', 'path': 'Human2/img', 'startFrame': 617, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-7', 'path': 'Human2/img', 'startFrame': 729, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-8', 'path': 'Human2/img', 'startFrame': 841, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human2-9', 'path': 'Human2/img', 'startFrame': 953, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-0', 'path': 'Human3/img', 'startFrame': 1, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-1', 'path': 'Human3/img', 'startFrame': 136, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-2', 'path': 'Human3/img', 'startFrame': 220, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-3', 'path': 'Human3/img', 'startFrame': 389, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-4', 'path': 'Human3/img', 'startFrame': 541, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-5', 'path': 'Human3/img', 'startFrame': 693, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-6', 'path': 'Human3/img', 'startFrame': 947, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-7', 'path': 'Human3/img', 'startFrame': 1133, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-8', 'path': 'Human3/img', 'startFrame': 1234, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human3-9', 'path': 'Human3/img', 'startFrame': 1437, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-0', 'path': 'Human4/img', 'startFrame': 1, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-1', 'path': 'Human4/img', 'startFrame': 67, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-2', 'path': 'Human4/img', 'startFrame': 113, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-3', 'path': 'Human4/img', 'startFrame': 146, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-4', 'path': 'Human4/img', 'startFrame': 212, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-5', 'path': 'Human4/img', 'startFrame': 304, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-6', 'path': 'Human4/img', 'startFrame': 390, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-7', 'path': 'Human4/img', 'startFrame': 436, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-8', 'path': 'Human4/img', 'startFrame': 509, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human4_2-9', 'path': 'Human4/img', 'startFrame': 562, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-0', 'path': 'Human5/img', 'startFrame': 1, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-1', 'path': 'Human5/img', 'startFrame': 22, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-2', 'path': 'Human5/img', 'startFrame': 86, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-3', 'path': 'Human5/img', 'startFrame': 171, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-4', 'path': 'Human5/img', 'startFrame': 242, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-5', 'path': 'Human5/img', 'startFrame': 327, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-6', 'path': 'Human5/img', 'startFrame': 370, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-7', 'path': 'Human5/img', 'startFrame': 448, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-8', 'path': 'Human5/img', 'startFrame': 547, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human5-9', 'path': 'Human5/img', 'startFrame': 590, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-0', 'path': 'Human6/img', 'startFrame': 1, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-1', 'path': 'Human6/img', 'startFrame': 48, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-2', 'path': 'Human6/img', 'startFrame': 127, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-3', 'path': 'Human6/img', 'startFrame': 182, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-4', 'path': 'Human6/img', 'startFrame': 277, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-5', 'path': 'Human6/img', 'startFrame': 332, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-6', 'path': 'Human6/img', 'startFrame': 403, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-7', 'path': 'Human6/img', 'startFrame': 514, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-8', 'path': 'Human6/img', 'startFrame': 617, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human6-9', 'path': 'Human6/img', 'startFrame': 696, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-0', 'path': 'Human7/img', 'startFrame': 1, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-1', 'path': 'Human7/img', 'startFrame': 8, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-2', 'path': 'Human7/img', 'startFrame': 48, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-3', 'path': 'Human7/img', 'startFrame': 58, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-4', 'path': 'Human7/img', 'startFrame': 86, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-5', 'path': 'Human7/img', 'startFrame': 111, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-6', 'path': 'Human7/img', 'startFrame': 136, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-7', 'path': 'Human7/img', 'startFrame': 171, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-8', 'path': 'Human7/img', 'startFrame': 188, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human7-9', 'path': 'Human7/img', 'startFrame': 221, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-0', 'path': 'Human8/img', 'startFrame': 1, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-1', 'path': 'Human8/img', 'startFrame': 8, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-2', 'path': 'Human8/img', 'startFrame': 14, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-3', 'path': 'Human8/img', 'startFrame': 31, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-4', 'path': 'Human8/img', 'startFrame': 47, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-5', 'path': 'Human8/img', 'startFrame': 50, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-6', 'path': 'Human8/img', 'startFrame': 64, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-7', 'path': 'Human8/img', 'startFrame': 76, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-8', 'path': 'Human8/img', 'startFrame': 93, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human8-9', 'path': 'Human8/img', 'startFrame': 106, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-0', 'path': 'Human9/img', 'startFrame': 1, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-1', 'path': 'Human9/img', 'startFrame': 7, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-2', 'path': 'Human9/img', 'startFrame': 58, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-3', 'path': 'Human9/img', 'startFrame': 73, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-4', 'path': 'Human9/img', 'startFrame': 103, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-5', 'path': 'Human9/img', 'startFrame': 145, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-6', 'path': 'Human9/img', 'startFrame': 181, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-7', 'path': 'Human9/img', 'startFrame': 196, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-8', 'path': 'Human9/img', 'startFrame': 226, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Human9-9', 'path': 'Human9/img', 'startFrame': 259, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Ironman-0', 'path': 'Ironman/img', 'startFrame': 1, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-1', 'path': 'Ironman/img', 'startFrame': 15, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-2', 'path': 'Ironman/img', 'startFrame': 23, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-3', 'path': 'Ironman/img', 'startFrame': 34, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-4', 'path': 'Ironman/img', 'startFrame': 60, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-5', 'path': 'Ironman/img', 'startFrame': 69, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-6', 'path': 'Ironman/img', 'startFrame': 85, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-7', 'path': 'Ironman/img', 'startFrame': 103, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-8', 'path': 'Ironman/img', 'startFrame': 122, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Ironman-9', 'path': 'Ironman/img', 'startFrame': 132, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Jogging_1-0', 'path': 'Jogging/img', 'startFrame': 1, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-1', 'path': 'Jogging/img', 'startFrame': 22, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-2', 'path': 'Jogging/img', 'startFrame': 37, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-3', 'path': 'Jogging/img', 'startFrame': 76, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-4', 'path': 'Jogging/img', 'startFrame': 97, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-5', 'path': 'Jogging/img', 'startFrame': 133, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-6', 'path': 'Jogging/img', 'startFrame': 178, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-7', 'path': 'Jogging/img', 'startFrame': 211, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-8', 'path': 'Jogging/img', 'startFrame': 217, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_1-9', 'path': 'Jogging/img', 'startFrame': 262, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-0', 'path': 'Jogging/img', 'startFrame': 1, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-1', 'path': 'Jogging/img', 'startFrame': 16, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-2', 'path': 'Jogging/img', 'startFrame': 52, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-3', 'path': 'Jogging/img', 'startFrame': 82, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-4', 'path': 'Jogging/img', 'startFrame': 103, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-5', 'path': 'Jogging/img', 'startFrame': 148, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-6', 'path': 'Jogging/img', 'startFrame': 181, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-7', 'path': 'Jogging/img', 'startFrame': 202, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-8', 'path': 'Jogging/img', 'startFrame': 226, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jogging_2-9', 'path': 'Jogging/img', 'startFrame': 265, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-0', 'path': 'Jump/img', 'startFrame': 1, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-1', 'path': 'Jump/img', 'startFrame': 13, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-2', 'path': 'Jump/img', 'startFrame': 15, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-3', 'path': 'Jump/img', 'startFrame': 32, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-4', 'path': 'Jump/img', 'startFrame': 41, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-5', 'path': 'Jump/img', 'startFrame': 52, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-6', 'path': 'Jump/img', 'startFrame': 69, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-7', 'path': 'Jump/img', 'startFrame': 75, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-8', 'path': 'Jump/img', 'startFrame': 97, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jump-9', 'path': 'Jump/img', 'startFrame': 104, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Jumping-0', 'path': 'Jumping/img', 'startFrame': 1, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-1', 'path': 'Jumping/img', 'startFrame': 7, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-2', 'path': 'Jumping/img', 'startFrame': 41, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-3', 'path': 'Jumping/img', 'startFrame': 87, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-4', 'path': 'Jumping/img', 'startFrame': 121, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-5', 'path': 'Jumping/img', 'startFrame': 156, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-6', 'path': 'Jumping/img', 'startFrame': 174, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-7', 'path': 'Jumping/img', 'startFrame': 211, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-8', 'path': 'Jumping/img', 'startFrame': 245, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Jumping-9', 'path': 'Jumping/img', 'startFrame': 261, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-0', 'path': 'KiteSurf/img', 'startFrame': 1, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-1', 'path': 'KiteSurf/img', 'startFrame': 2, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-2', 'path': 'KiteSurf/img', 'startFrame': 11, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-3', 'path': 'KiteSurf/img', 'startFrame': 20, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-4', 'path': 'KiteSurf/img', 'startFrame': 32, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-5', 'path': 'KiteSurf/img', 'startFrame': 38, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-6', 'path': 'KiteSurf/img', 'startFrame': 44, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-7', 'path': 'KiteSurf/img', 'startFrame': 52, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-8', 'path': 'KiteSurf/img', 'startFrame': 63, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'KiteSurf-9', 'path': 'KiteSurf/img', 'startFrame': 67, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Lemming-0', 'path': 'Lemming/img', 'startFrame': 1, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-1', 'path': 'Lemming/img', 'startFrame': 134, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-2', 'path': 'Lemming/img', 'startFrame': 173, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-3', 'path': 'Lemming/img', 'startFrame': 293, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-4', 'path': 'Lemming/img', 'startFrame': 466, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-5', 'path': 'Lemming/img', 'startFrame': 559, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-6', 'path': 'Lemming/img', 'startFrame': 679, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-7', 'path': 'Lemming/img', 'startFrame': 865, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-8', 'path': 'Lemming/img', 'startFrame': 1025, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Lemming-9', 'path': 'Lemming/img', 'startFrame': 1131, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-0', 'path': 'Liquor/img', 'startFrame': 1, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-1', 'path': 'Liquor/img', 'startFrame': 105, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-2', 'path': 'Liquor/img', 'startFrame': 262, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-3', 'path': 'Liquor/img', 'startFrame': 418, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-4', 'path': 'Liquor/img', 'startFrame': 662, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-5', 'path': 'Liquor/img', 'startFrame': 784, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-6', 'path': 'Liquor/img', 'startFrame': 1010, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-7', 'path': 'Liquor/img', 'startFrame': 1201, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-8', 'path': 'Liquor/img', 'startFrame': 1253, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Liquor-9', 'path': 'Liquor/img', 'startFrame': 1445, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Man-0', 'path': 'Man/img', 'startFrame': 1, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-1', 'path': 'Man/img', 'startFrame': 4, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-2', 'path': 'Man/img', 'startFrame': 24, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-3', 'path': 'Man/img', 'startFrame': 29, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-4', 'path': 'Man/img', 'startFrame': 46, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-5', 'path': 'Man/img', 'startFrame': 66, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-6', 'path': 'Man/img', 'startFrame': 67, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-7', 'path': 'Man/img', 'startFrame': 88, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-8', 'path': 'Man/img', 'startFrame': 101, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Man-9', 'path': 'Man/img', 'startFrame': 110, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Matrix-0', 'path': 'Matrix/img', 'startFrame': 1, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-1', 'path': 'Matrix/img', 'startFrame': 4, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-2', 'path': 'Matrix/img', 'startFrame': 12, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-3', 'path': 'Matrix/img', 'startFrame': 31, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-4', 'path': 'Matrix/img', 'startFrame': 35, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-5', 'path': 'Matrix/img', 'startFrame': 49, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-6', 'path': 'Matrix/img', 'startFrame': 60, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-7', 'path': 'Matrix/img', 'startFrame': 64, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-8', 'path': 'Matrix/img', 'startFrame': 75, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Matrix-9', 'path': 'Matrix/img', 'startFrame': 83, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Mhyang-0', 'path': 'Mhyang/img', 'startFrame': 1, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-1', 'path': 'Mhyang/img', 'startFrame': 45, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-2', 'path': 'Mhyang/img', 'startFrame': 194, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-3', 'path': 'Mhyang/img', 'startFrame': 448, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-4', 'path': 'Mhyang/img', 'startFrame': 507, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-5', 'path': 'Mhyang/img', 'startFrame': 641, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-6', 'path': 'Mhyang/img', 'startFrame': 835, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-7', 'path': 'Mhyang/img', 'startFrame': 1044, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-8', 'path': 'Mhyang/img', 'startFrame': 1088, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Mhyang-9', 'path': 'Mhyang/img', 'startFrame': 1312, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'MotorRolling-0', 'path': 'MotorRolling/img', 'startFrame': 1, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-1', 'path': 'MotorRolling/img', 'startFrame': 4, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-2', 'path': 'MotorRolling/img', 'startFrame': 23, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-3', 'path': 'MotorRolling/img', 'startFrame': 41, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-4', 'path': 'MotorRolling/img', 'startFrame': 55, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-5', 'path': 'MotorRolling/img', 'startFrame': 77, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-6', 'path': 'MotorRolling/img', 'startFrame': 84, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-7', 'path': 'MotorRolling/img', 'startFrame': 100, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-8', 'path': 'MotorRolling/img', 'startFrame': 116, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MotorRolling-9', 'path': 'MotorRolling/img', 'startFrame': 141, 'endFrame': 164, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'MountainBike-0', 'path': 'MountainBike/img', 'startFrame': 1, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-1', 'path': 'MountainBike/img', 'startFrame': 20, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-2', 'path': 'MountainBike/img', 'startFrame': 45, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-3', 'path': 'MountainBike/img', 'startFrame': 62, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-4', 'path': 'MountainBike/img', 'startFrame': 89, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-5', 'path': 'MountainBike/img', 'startFrame': 106, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-6', 'path': 'MountainBike/img', 'startFrame': 119, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-7', 'path': 'MountainBike/img', 'startFrame': 144, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-8', 'path': 'MountainBike/img', 'startFrame': 163, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'MountainBike-9', 'path': 'MountainBike/img', 'startFrame': 194, 'endFrame': 228, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
            ,
            {'name': 'Panda-0', 'path': 'Panda/img', 'startFrame': 1, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-1', 'path': 'Panda/img', 'startFrame': 61, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-2', 'path': 'Panda/img', 'startFrame': 201, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-3', 'path': 'Panda/img', 'startFrame': 241, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-4', 'path': 'Panda/img', 'startFrame': 311, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-5', 'path': 'Panda/img', 'startFrame': 501, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-6', 'path': 'Panda/img', 'startFrame': 541, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-7', 'path': 'Panda/img', 'startFrame': 701, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-8', 'path': 'Panda/img', 'startFrame': 771, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'Panda-9', 'path': 'Panda/img', 'startFrame': 841, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
            ,
            {'name': 'RedTeam-0', 'path': 'RedTeam/img', 'startFrame': 1, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-1', 'path': 'RedTeam/img', 'startFrame': 58, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-2', 'path': 'RedTeam/img', 'startFrame': 325, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-3', 'path': 'RedTeam/img', 'startFrame': 574, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-4', 'path': 'RedTeam/img', 'startFrame': 765, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-5', 'path': 'RedTeam/img', 'startFrame': 803, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-6', 'path': 'RedTeam/img', 'startFrame': 1127, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-7', 'path': 'RedTeam/img', 'startFrame': 1204, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-8', 'path': 'RedTeam/img', 'startFrame': 1490, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'RedTeam-9', 'path': 'RedTeam/img', 'startFrame': 1605, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
            ,
            {'name': 'Rubik-0', 'path': 'Rubik/img', 'startFrame': 1, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-1', 'path': 'Rubik/img', 'startFrame': 60, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-2', 'path': 'Rubik/img', 'startFrame': 219, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-3', 'path': 'Rubik/img', 'startFrame': 518, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-4', 'path': 'Rubik/img', 'startFrame': 617, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-5', 'path': 'Rubik/img', 'startFrame': 896, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-6', 'path': 'Rubik/img', 'startFrame': 1015, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-7', 'path': 'Rubik/img', 'startFrame': 1274, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-8', 'path': 'Rubik/img', 'startFrame': 1493, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Rubik-9', 'path': 'Rubik/img', 'startFrame': 1612, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Shaking-0', 'path': 'Shaking/img', 'startFrame': 1, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-1', 'path': 'Shaking/img', 'startFrame': 8, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-2', 'path': 'Shaking/img', 'startFrame': 65, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-3', 'path': 'Shaking/img', 'startFrame': 109, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-4', 'path': 'Shaking/img', 'startFrame': 145, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-5', 'path': 'Shaking/img', 'startFrame': 173, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-6', 'path': 'Shaking/img', 'startFrame': 217, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-7', 'path': 'Shaking/img', 'startFrame': 242, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-8', 'path': 'Shaking/img', 'startFrame': 285, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Shaking-9', 'path': 'Shaking/img', 'startFrame': 325, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Singer1-0', 'path': 'Singer1/img', 'startFrame': 1, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-1', 'path': 'Singer1/img', 'startFrame': 15, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-2', 'path': 'Singer1/img', 'startFrame': 43, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-3', 'path': 'Singer1/img', 'startFrame': 81, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-4', 'path': 'Singer1/img', 'startFrame': 130, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-5', 'path': 'Singer1/img', 'startFrame': 144, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-6', 'path': 'Singer1/img', 'startFrame': 207, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-7', 'path': 'Singer1/img', 'startFrame': 225, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-8', 'path': 'Singer1/img', 'startFrame': 277, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer1-9', 'path': 'Singer1/img', 'startFrame': 288, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-0', 'path': 'Singer2/img', 'startFrame': 1, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-1', 'path': 'Singer2/img', 'startFrame': 19, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-2', 'path': 'Singer2/img', 'startFrame': 58, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-3', 'path': 'Singer2/img', 'startFrame': 83, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-4', 'path': 'Singer2/img', 'startFrame': 130, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-5', 'path': 'Singer2/img', 'startFrame': 159, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-6', 'path': 'Singer2/img', 'startFrame': 199, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-7', 'path': 'Singer2/img', 'startFrame': 253, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-8', 'path': 'Singer2/img', 'startFrame': 260, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Singer2-9', 'path': 'Singer2/img', 'startFrame': 310, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-0', 'path': 'Skater/img', 'startFrame': 1, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-1', 'path': 'Skater/img', 'startFrame': 2, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-2', 'path': 'Skater/img', 'startFrame': 26, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-3', 'path': 'Skater/img', 'startFrame': 39, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-4', 'path': 'Skater/img', 'startFrame': 58, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-5', 'path': 'Skater/img', 'startFrame': 76, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-6', 'path': 'Skater/img', 'startFrame': 84, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-7', 'path': 'Skater/img', 'startFrame': 105, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-8', 'path': 'Skater/img', 'startFrame': 114, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater-9', 'path': 'Skater/img', 'startFrame': 141, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-0', 'path': 'Skater2/img', 'startFrame': 1, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-1', 'path': 'Skater2/img', 'startFrame': 18, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-2', 'path': 'Skater2/img', 'startFrame': 65, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-3', 'path': 'Skater2/img', 'startFrame': 108, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-4', 'path': 'Skater2/img', 'startFrame': 142, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-5', 'path': 'Skater2/img', 'startFrame': 203, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-6', 'path': 'Skater2/img', 'startFrame': 224, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-7', 'path': 'Skater2/img', 'startFrame': 263, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-8', 'path': 'Skater2/img', 'startFrame': 327, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skater2-9', 'path': 'Skater2/img', 'startFrame': 357, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-0', 'path': 'Skating1/img', 'startFrame': 1, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-1', 'path': 'Skating1/img', 'startFrame': 13, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-2', 'path': 'Skating1/img', 'startFrame': 57, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-3', 'path': 'Skating1/img', 'startFrame': 93, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-4', 'path': 'Skating1/img', 'startFrame': 141, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-5', 'path': 'Skating1/img', 'startFrame': 173, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-6', 'path': 'Skating1/img', 'startFrame': 217, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-7', 'path': 'Skating1/img', 'startFrame': 269, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-8', 'path': 'Skating1/img', 'startFrame': 317, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating1-9', 'path': 'Skating1/img', 'startFrame': 345, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-0', 'path': 'Skating2/img', 'startFrame': 1, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-1', 'path': 'Skating2/img', 'startFrame': 43, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-2', 'path': 'Skating2/img', 'startFrame': 76, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-3', 'path': 'Skating2/img', 'startFrame': 99, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-4', 'path': 'Skating2/img', 'startFrame': 165, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-5', 'path': 'Skating2/img', 'startFrame': 203, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-6', 'path': 'Skating2/img', 'startFrame': 283, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-7', 'path': 'Skating2/img', 'startFrame': 330, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-8', 'path': 'Skating2/img', 'startFrame': 348, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_1-9', 'path': 'Skating2/img', 'startFrame': 424, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-0', 'path': 'Skating2/img', 'startFrame': 1, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-1', 'path': 'Skating2/img', 'startFrame': 19, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-2', 'path': 'Skating2/img', 'startFrame': 80, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-3', 'path': 'Skating2/img', 'startFrame': 142, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-4', 'path': 'Skating2/img', 'startFrame': 184, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-5', 'path': 'Skating2/img', 'startFrame': 212, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-6', 'path': 'Skating2/img', 'startFrame': 254, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-7', 'path': 'Skating2/img', 'startFrame': 297, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-8', 'path': 'Skating2/img', 'startFrame': 367, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skating2_2-9', 'path': 'Skating2/img', 'startFrame': 386, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-0', 'path': 'Skiing/img', 'startFrame': 1, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-1', 'path': 'Skiing/img', 'startFrame': 9, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-2', 'path': 'Skiing/img', 'startFrame': 15, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-3', 'path': 'Skiing/img', 'startFrame': 23, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-4', 'path': 'Skiing/img', 'startFrame': 29, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-5', 'path': 'Skiing/img', 'startFrame': 35, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-6', 'path': 'Skiing/img', 'startFrame': 45, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-7', 'path': 'Skiing/img', 'startFrame': 53, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-8', 'path': 'Skiing/img', 'startFrame': 57, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Skiing-9', 'path': 'Skiing/img', 'startFrame': 68, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Soccer-0', 'path': 'Soccer/img', 'startFrame': 1, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-1', 'path': 'Soccer/img', 'startFrame': 24, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-2', 'path': 'Soccer/img', 'startFrame': 67, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-3', 'path': 'Soccer/img', 'startFrame': 98, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-4', 'path': 'Soccer/img', 'startFrame': 141, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-5', 'path': 'Soccer/img', 'startFrame': 168, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-6', 'path': 'Soccer/img', 'startFrame': 207, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-7', 'path': 'Soccer/img', 'startFrame': 238, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-8', 'path': 'Soccer/img', 'startFrame': 293, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Soccer-9', 'path': 'Soccer/img', 'startFrame': 336, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Subway-0', 'path': 'Subway/img', 'startFrame': 1, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-1', 'path': 'Subway/img', 'startFrame': 11, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-2', 'path': 'Subway/img', 'startFrame': 23, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-3', 'path': 'Subway/img', 'startFrame': 36, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-4', 'path': 'Subway/img', 'startFrame': 55, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-5', 'path': 'Subway/img', 'startFrame': 80, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-6', 'path': 'Subway/img', 'startFrame': 91, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-7', 'path': 'Subway/img', 'startFrame': 114, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-8', 'path': 'Subway/img', 'startFrame': 133, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Subway-9', 'path': 'Subway/img', 'startFrame': 152, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Surfer-0', 'path': 'Surfer/img', 'startFrame': 1, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-1', 'path': 'Surfer/img', 'startFrame': 34, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-2', 'path': 'Surfer/img', 'startFrame': 56, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-3', 'path': 'Surfer/img', 'startFrame': 108, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-4', 'path': 'Surfer/img', 'startFrame': 115, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-5', 'path': 'Surfer/img', 'startFrame': 167, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-6', 'path': 'Surfer/img', 'startFrame': 223, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-7', 'path': 'Surfer/img', 'startFrame': 245, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-8', 'path': 'Surfer/img', 'startFrame': 271, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Surfer-9', 'path': 'Surfer/img', 'startFrame': 322, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
            ,
            {'name': 'Suv-0', 'path': 'Suv/img', 'startFrame': 1, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-1', 'path': 'Suv/img', 'startFrame': 48, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-2', 'path': 'Suv/img', 'startFrame': 113, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-3', 'path': 'Suv/img', 'startFrame': 264, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-4', 'path': 'Suv/img', 'startFrame': 311, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-5', 'path': 'Suv/img', 'startFrame': 471, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-6', 'path': 'Suv/img', 'startFrame': 565, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-7', 'path': 'Suv/img', 'startFrame': 593, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-8', 'path': 'Suv/img', 'startFrame': 677, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Suv-9', 'path': 'Suv/img', 'startFrame': 847, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
            ,
            {'name': 'Sylvester-0', 'path': 'Sylvester/img', 'startFrame': 1, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-1', 'path': 'Sylvester/img', 'startFrame': 27, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-2', 'path': 'Sylvester/img', 'startFrame': 175, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-3', 'path': 'Sylvester/img', 'startFrame': 309, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-4', 'path': 'Sylvester/img', 'startFrame': 470, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-5', 'path': 'Sylvester/img', 'startFrame': 590, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-6', 'path': 'Sylvester/img', 'startFrame': 778, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-7', 'path': 'Sylvester/img', 'startFrame': 818, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-8', 'path': 'Sylvester/img', 'startFrame': 1006, 'endFrame': 1345, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Sylvester-9', 'path': 'Sylvester/img', 'startFrame': 1207, 'endFrame': 1345, 'nz': 4,
             'ext': 'jpg', 'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger1-0', 'path': 'Tiger1/img', 'startFrame': 1, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-1', 'path': 'Tiger1/img', 'startFrame': 11, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-2', 'path': 'Tiger1/img', 'startFrame': 67, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-3', 'path': 'Tiger1/img', 'startFrame': 92, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-4', 'path': 'Tiger1/img', 'startFrame': 113, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-5', 'path': 'Tiger1/img', 'startFrame': 169, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-6', 'path': 'Tiger1/img', 'startFrame': 190, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-7', 'path': 'Tiger1/img', 'startFrame': 246, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-8', 'path': 'Tiger1/img', 'startFrame': 267, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger1-9', 'path': 'Tiger1/img', 'startFrame': 312, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
            ,
            {'name': 'Tiger2-0', 'path': 'Tiger2/img', 'startFrame': 1, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-1', 'path': 'Tiger2/img', 'startFrame': 22, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-2', 'path': 'Tiger2/img', 'startFrame': 40, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-3', 'path': 'Tiger2/img', 'startFrame': 76, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-4', 'path': 'Tiger2/img', 'startFrame': 141, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-5', 'path': 'Tiger2/img', 'startFrame': 170, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-6', 'path': 'Tiger2/img', 'startFrame': 213, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-7', 'path': 'Tiger2/img', 'startFrame': 242, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-8', 'path': 'Tiger2/img', 'startFrame': 281, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Tiger2-9', 'path': 'Tiger2/img', 'startFrame': 321, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-0', 'path': 'Toy/img', 'startFrame': 1, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-1', 'path': 'Toy/img', 'startFrame': 11, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-2', 'path': 'Toy/img', 'startFrame': 33, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-3', 'path': 'Toy/img', 'startFrame': 76, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-4', 'path': 'Toy/img', 'startFrame': 98, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-5', 'path': 'Toy/img', 'startFrame': 133, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-6', 'path': 'Toy/img', 'startFrame': 154, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-7', 'path': 'Toy/img', 'startFrame': 176, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-8', 'path': 'Toy/img', 'startFrame': 198, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Toy-9', 'path': 'Toy/img', 'startFrame': 219, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-0', 'path': 'Trans/img', 'startFrame': 1, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-1', 'path': 'Trans/img', 'startFrame': 3, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-2', 'path': 'Trans/img', 'startFrame': 19, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-3', 'path': 'Trans/img', 'startFrame': 28, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-4', 'path': 'Trans/img', 'startFrame': 40, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-5', 'path': 'Trans/img', 'startFrame': 55, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-6', 'path': 'Trans/img', 'startFrame': 63, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-7', 'path': 'Trans/img', 'startFrame': 81, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-8', 'path': 'Trans/img', 'startFrame': 86, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trans-9', 'path': 'Trans/img', 'startFrame': 101, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Trellis-0', 'path': 'Trellis/img', 'startFrame': 1, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-1', 'path': 'Trellis/img', 'startFrame': 12, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-2', 'path': 'Trellis/img', 'startFrame': 73, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-3', 'path': 'Trellis/img', 'startFrame': 124, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-4', 'path': 'Trellis/img', 'startFrame': 202, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-5', 'path': 'Trellis/img', 'startFrame': 236, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-6', 'path': 'Trellis/img', 'startFrame': 303, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-7', 'path': 'Trellis/img', 'startFrame': 359, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-8', 'path': 'Trellis/img', 'startFrame': 443, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Trellis-9', 'path': 'Trellis/img', 'startFrame': 471, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
            ,
            {'name': 'Twinnings-0', 'path': 'Twinnings/img', 'startFrame': 1, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-1', 'path': 'Twinnings/img', 'startFrame': 15, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-2', 'path': 'Twinnings/img', 'startFrame': 57, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-3', 'path': 'Twinnings/img', 'startFrame': 137, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-4', 'path': 'Twinnings/img', 'startFrame': 165, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-5', 'path': 'Twinnings/img', 'startFrame': 217, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-6', 'path': 'Twinnings/img', 'startFrame': 268, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-7', 'path': 'Twinnings/img', 'startFrame': 301, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-8', 'path': 'Twinnings/img', 'startFrame': 372, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Twinnings-9', 'path': 'Twinnings/img', 'startFrame': 400, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-0', 'path': 'Vase/img', 'startFrame': 1, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-1', 'path': 'Vase/img', 'startFrame': 19, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-2', 'path': 'Vase/img', 'startFrame': 49, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-3', 'path': 'Vase/img', 'startFrame': 71, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-4', 'path': 'Vase/img', 'startFrame': 103, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-5', 'path': 'Vase/img', 'startFrame': 122, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-6', 'path': 'Vase/img', 'startFrame': 138, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-7', 'path': 'Vase/img', 'startFrame': 171, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-8', 'path': 'Vase/img', 'startFrame': 206, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Vase-9', 'path': 'Vase/img', 'startFrame': 219, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
            ,
            {'name': 'Walking-0', 'path': 'Walking/img', 'startFrame': 1, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-1', 'path': 'Walking/img', 'startFrame': 21, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-2', 'path': 'Walking/img', 'startFrame': 78, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-3', 'path': 'Walking/img', 'startFrame': 107, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-4', 'path': 'Walking/img', 'startFrame': 160, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-5', 'path': 'Walking/img', 'startFrame': 197, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-6', 'path': 'Walking/img', 'startFrame': 222, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-7', 'path': 'Walking/img', 'startFrame': 259, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-8', 'path': 'Walking/img', 'startFrame': 324, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking-9', 'path': 'Walking/img', 'startFrame': 353, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-0', 'path': 'Walking2/img', 'startFrame': 1, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-1', 'path': 'Walking2/img', 'startFrame': 26, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-2', 'path': 'Walking2/img', 'startFrame': 71, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-3', 'path': 'Walking2/img', 'startFrame': 146, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-4', 'path': 'Walking2/img', 'startFrame': 171, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-5', 'path': 'Walking2/img', 'startFrame': 246, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-6', 'path': 'Walking2/img', 'startFrame': 271, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-7', 'path': 'Walking2/img', 'startFrame': 336, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-8', 'path': 'Walking2/img', 'startFrame': 396, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Walking2-9', 'path': 'Walking2/img', 'startFrame': 446, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-0', 'path': 'Woman/img', 'startFrame': 1, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-1', 'path': 'Woman/img', 'startFrame': 6, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-2', 'path': 'Woman/img', 'startFrame': 119, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-3', 'path': 'Woman/img', 'startFrame': 160, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-4', 'path': 'Woman/img', 'startFrame': 219, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-5', 'path': 'Woman/img', 'startFrame': 272, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-6', 'path': 'Woman/img', 'startFrame': 343, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-7', 'path': 'Woman/img', 'startFrame': 402, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-8', 'path': 'Woman/img', 'startFrame': 449, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
            {'name': 'Woman-9', 'path': 'Woman/img', 'startFrame': 520, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
             'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
            ,
        ]

        return sequence_info_list




# import numpy as np
# from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
# from lib.test.utils.load_text import load_text
#
#
# class OTBDataset(BaseDataset):
#     """ OTB-2015 dataset
#     Publication:
#         Object Tracking Benchmark
#         Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
#         TPAMI, 2015
#         http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
#     Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.base_path = self.env_settings.otb_path
#         self.sequence_info_list = self._get_sequence_info_list()
#
#     def get_sequence_list(self):
#         return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])
#
#     def _construct_sequence(self, sequence_info):
#         sequence_path = sequence_info['path']
#         nz = sequence_info['nz']
#         ext = sequence_info['ext']
#         start_frame = sequence_info['startFrame']
#         end_frame = sequence_info['endFrame']
#
#         init_omit = 0
#         if 'initOmit' in sequence_info:
#             init_omit = sequence_info['initOmit']
#
#         frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
#                                                                            sequence_path=sequence_path, frame=frame_num,
#                                                                            nz=nz, ext=ext) for frame_num in
#                   range(start_frame + init_omit, end_frame + 1)]
#
#         # anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])
#         anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])
#
#         # NOTE: OTB has some weird annos which panda cannot handle
#         ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
#
#         return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:, :],
#                         object_class=sequence_info['object_class'])
#
#     def __len__(self):
#         return len(self.sequence_info_list)
#
#     def _get_sequence_info_list(self):
#         sequence_info_list = [
#             {'name': 'Basketball-0', 'path': 'Basketball/img', 'startFrame': 1, 'endFrame': 725, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-1', 'path': 'Basketball/img', 'startFrame': 37, 'endFrame': 725, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-2', 'path': 'Basketball/img', 'startFrame': 58, 'endFrame': 725, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-3', 'path': 'Basketball/img', 'startFrame': 44, 'endFrame': 725, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-4', 'path': 'Basketball/img', 'startFrame': 173, 'endFrame': 725, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-5', 'path': 'Basketball/img', 'startFrame': 145, 'endFrame': 725, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-6', 'path': 'Basketball/img', 'startFrame': 346, 'endFrame': 725, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-7', 'path': 'Basketball/img', 'startFrame': 404, 'endFrame': 725, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-8', 'path': 'Basketball/img', 'startFrame': 461, 'endFrame': 725, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Basketball-9', 'path': 'Basketball/img', 'startFrame': 584, 'endFrame': 725, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'Basketball/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Biker-0', 'path': 'Biker/img', 'startFrame': 1, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-1', 'path': 'Biker/img', 'startFrame': 2, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-2', 'path': 'Biker/img', 'startFrame': 29, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-3', 'path': 'Biker/img', 'startFrame': 30, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-4', 'path': 'Biker/img', 'startFrame': 51, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-5', 'path': 'Biker/img', 'startFrame': 15, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-6', 'path': 'Biker/img', 'startFrame': 26, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-7', 'path': 'Biker/img', 'startFrame': 79, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-8', 'path': 'Biker/img', 'startFrame': 68, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Biker-9', 'path': 'Biker/img', 'startFrame': 13, 'endFrame': 142, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Biker/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Bird1-0', 'path': 'Bird1/img', 'startFrame': 1, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-1', 'path': 'Bird1/img', 'startFrame': 21, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-2', 'path': 'Bird1/img', 'startFrame': 17, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-3', 'path': 'Bird1/img', 'startFrame': 109, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-4', 'path': 'Bird1/img', 'startFrame': 49, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-5', 'path': 'Bird1/img', 'startFrame': 81, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-6', 'path': 'Bird1/img', 'startFrame': 145, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-7', 'path': 'Bird1/img', 'startFrame': 85, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-8', 'path': 'Bird1/img', 'startFrame': 257, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird1-9', 'path': 'Bird1/img', 'startFrame': 289, 'endFrame': 408, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird1/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-0', 'path': 'Bird2/img', 'startFrame': 1, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-1', 'path': 'Bird2/img', 'startFrame': 9, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-2', 'path': 'Bird2/img', 'startFrame': 2, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-3', 'path': 'Bird2/img', 'startFrame': 11, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-4', 'path': 'Bird2/img', 'startFrame': 8, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-5', 'path': 'Bird2/img', 'startFrame': 23, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-6', 'path': 'Bird2/img', 'startFrame': 11, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-7', 'path': 'Bird2/img', 'startFrame': 13, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-8', 'path': 'Bird2/img', 'startFrame': 51, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'Bird2-9', 'path': 'Bird2/img', 'startFrame': 57, 'endFrame': 99, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bird2/groundtruth_rect.txt', 'object_class': 'bird'}
#             ,
#             {'name': 'BlurBody-0', 'path': 'BlurBody/img', 'startFrame': 1, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-1', 'path': 'BlurBody/img', 'startFrame': 30, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-2', 'path': 'BlurBody/img', 'startFrame': 53, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-3', 'path': 'BlurBody/img', 'startFrame': 40, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-4', 'path': 'BlurBody/img', 'startFrame': 40, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-5', 'path': 'BlurBody/img', 'startFrame': 149, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-6', 'path': 'BlurBody/img', 'startFrame': 60, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-7', 'path': 'BlurBody/img', 'startFrame': 139, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-8', 'path': 'BlurBody/img', 'startFrame': 212, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurBody-9', 'path': 'BlurBody/img', 'startFrame': 30, 'endFrame': 334, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurBody/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'BlurCar1-0', 'path': 'BlurCar1/img', 'startFrame': 247, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-1', 'path': 'BlurCar1/img', 'startFrame': 261, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-2', 'path': 'BlurCar1/img', 'startFrame': 305, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-3', 'path': 'BlurCar1/img', 'startFrame': 335, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-4', 'path': 'BlurCar1/img', 'startFrame': 364, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-5', 'path': 'BlurCar1/img', 'startFrame': 492, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-6', 'path': 'BlurCar1/img', 'startFrame': 511, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-7', 'path': 'BlurCar1/img', 'startFrame': 281, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-8', 'path': 'BlurCar1/img', 'startFrame': 325, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar1-9', 'path': 'BlurCar1/img', 'startFrame': 467, 'endFrame': 988, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-0', 'path': 'BlurCar2/img', 'startFrame': 1, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-1', 'path': 'BlurCar2/img', 'startFrame': 53, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-2', 'path': 'BlurCar2/img', 'startFrame': 12, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-3', 'path': 'BlurCar2/img', 'startFrame': 175, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-4', 'path': 'BlurCar2/img', 'startFrame': 70, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-5', 'path': 'BlurCar2/img', 'startFrame': 175, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-6', 'path': 'BlurCar2/img', 'startFrame': 209, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-7', 'path': 'BlurCar2/img', 'startFrame': 204, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-8', 'path': 'BlurCar2/img', 'startFrame': 186, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar2-9', 'path': 'BlurCar2/img', 'startFrame': 105, 'endFrame': 585, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-0', 'path': 'BlurCar3/img', 'startFrame': 3, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-1', 'path': 'BlurCar3/img', 'startFrame': 24, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-2', 'path': 'BlurCar3/img', 'startFrame': 10, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-3', 'path': 'BlurCar3/img', 'startFrame': 87, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-4', 'path': 'BlurCar3/img', 'startFrame': 143, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-5', 'path': 'BlurCar3/img', 'startFrame': 125, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-6', 'path': 'BlurCar3/img', 'startFrame': 192, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-7', 'path': 'BlurCar3/img', 'startFrame': 52, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-8', 'path': 'BlurCar3/img', 'startFrame': 59, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar3-9', 'path': 'BlurCar3/img', 'startFrame': 34, 'endFrame': 359, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar3/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-0', 'path': 'BlurCar4/img', 'startFrame': 18, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-1', 'path': 'BlurCar4/img', 'startFrame': 32, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-2', 'path': 'BlurCar4/img', 'startFrame': 82, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-3', 'path': 'BlurCar4/img', 'startFrame': 61, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-4', 'path': 'BlurCar4/img', 'startFrame': 104, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-5', 'path': 'BlurCar4/img', 'startFrame': 162, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-6', 'path': 'BlurCar4/img', 'startFrame': 126, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-7', 'path': 'BlurCar4/img', 'startFrame': 68, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-8', 'path': 'BlurCar4/img', 'startFrame': 133, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurCar4-9', 'path': 'BlurCar4/img', 'startFrame': 180, 'endFrame': 397, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurCar4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'BlurFace-0', 'path': 'BlurFace/img', 'startFrame': 1, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-1', 'path': 'BlurFace/img', 'startFrame': 45, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-2', 'path': 'BlurFace/img', 'startFrame': 59, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-3', 'path': 'BlurFace/img', 'startFrame': 15, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-4', 'path': 'BlurFace/img', 'startFrame': 138, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-5', 'path': 'BlurFace/img', 'startFrame': 172, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-6', 'path': 'BlurFace/img', 'startFrame': 148, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-7', 'path': 'BlurFace/img', 'startFrame': 309, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-8', 'path': 'BlurFace/img', 'startFrame': 393, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurFace-9', 'path': 'BlurFace/img', 'startFrame': 397, 'endFrame': 493, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'BlurOwl-0', 'path': 'BlurOwl/img', 'startFrame': 1, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-1', 'path': 'BlurOwl/img', 'startFrame': 57, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-2', 'path': 'BlurOwl/img', 'startFrame': 114, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-3', 'path': 'BlurOwl/img', 'startFrame': 133, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-4', 'path': 'BlurOwl/img', 'startFrame': 51, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-5', 'path': 'BlurOwl/img', 'startFrame': 253, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-6', 'path': 'BlurOwl/img', 'startFrame': 227, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-7', 'path': 'BlurOwl/img', 'startFrame': 45, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-8', 'path': 'BlurOwl/img', 'startFrame': 202, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'BlurOwl-9', 'path': 'BlurOwl/img', 'startFrame': 227, 'endFrame': 631, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'BlurOwl/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-0', 'path': 'Board/img', 'startFrame': 1, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-1', 'path': 'Board/img', 'startFrame': 56, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-2', 'path': 'Board/img', 'startFrame': 42, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-3', 'path': 'Board/img', 'startFrame': 42, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-4', 'path': 'Board/img', 'startFrame': 277, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-5', 'path': 'Board/img', 'startFrame': 346, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-6', 'path': 'Board/img', 'startFrame': 332, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-7', 'path': 'Board/img', 'startFrame': 194, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-8', 'path': 'Board/img', 'startFrame': 497, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Board-9', 'path': 'Board/img', 'startFrame': 63, 'endFrame': 698, 'nz': 5, 'ext': 'jpg',
#              'anno_path': 'Board/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Bolt-0', 'path': 'Bolt/img', 'startFrame': 1, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-1', 'path': 'Bolt/img', 'startFrame': 29, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-2', 'path': 'Bolt/img', 'startFrame': 29, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-3', 'path': 'Bolt/img', 'startFrame': 95, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-4', 'path': 'Bolt/img', 'startFrame': 127, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-5', 'path': 'Bolt/img', 'startFrame': 88, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-6', 'path': 'Bolt/img', 'startFrame': 63, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-7', 'path': 'Bolt/img', 'startFrame': 99, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-8', 'path': 'Bolt/img', 'startFrame': 253, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt-9', 'path': 'Bolt/img', 'startFrame': 127, 'endFrame': 350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-0', 'path': 'Bolt2/img', 'startFrame': 1, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-1', 'path': 'Bolt2/img', 'startFrame': 12, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-2', 'path': 'Bolt2/img', 'startFrame': 41, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-3', 'path': 'Bolt2/img', 'startFrame': 35, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-4', 'path': 'Bolt2/img', 'startFrame': 117, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-5', 'path': 'Bolt2/img', 'startFrame': 30, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-6', 'path': 'Bolt2/img', 'startFrame': 88, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-7', 'path': 'Bolt2/img', 'startFrame': 41, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-8', 'path': 'Bolt2/img', 'startFrame': 117, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Bolt2-9', 'path': 'Bolt2/img', 'startFrame': 53, 'endFrame': 293, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Bolt2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Box-0', 'path': 'Box/img', 'startFrame': 1, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-1', 'path': 'Box/img', 'startFrame': 35, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-2', 'path': 'Box/img', 'startFrame': 209, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-3', 'path': 'Box/img', 'startFrame': 105, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-4', 'path': 'Box/img', 'startFrame': 93, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-5', 'path': 'Box/img', 'startFrame': 291, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-6', 'path': 'Box/img', 'startFrame': 279, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-7', 'path': 'Box/img', 'startFrame': 813, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-8', 'path': 'Box/img', 'startFrame': 557, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Box-9', 'path': 'Box/img', 'startFrame': 940, 'endFrame': 1161, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Box/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Boy-0', 'path': 'Boy/img', 'startFrame': 1, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-1', 'path': 'Boy/img', 'startFrame': 13, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-2', 'path': 'Boy/img', 'startFrame': 49, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-3', 'path': 'Boy/img', 'startFrame': 108, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-4', 'path': 'Boy/img', 'startFrame': 121, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-5', 'path': 'Boy/img', 'startFrame': 121, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-6', 'path': 'Boy/img', 'startFrame': 252, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-7', 'path': 'Boy/img', 'startFrame': 294, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-8', 'path': 'Boy/img', 'startFrame': 433, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Boy-9', 'path': 'Boy/img', 'startFrame': 109, 'endFrame': 602, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Boy/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Car1-0', 'path': 'Car1/img', 'startFrame': 1, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-1', 'path': 'Car1/img', 'startFrame': 82, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-2', 'path': 'Car1/img', 'startFrame': 41, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-3', 'path': 'Car1/img', 'startFrame': 215, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-4', 'path': 'Car1/img', 'startFrame': 205, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-5', 'path': 'Car1/img', 'startFrame': 256, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-6', 'path': 'Car1/img', 'startFrame': 368, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-7', 'path': 'Car1/img', 'startFrame': 500, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-8', 'path': 'Car1/img', 'startFrame': 490, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car1-9', 'path': 'Car1/img', 'startFrame': 184, 'endFrame': 1020, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car1/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-0', 'path': 'Car2/img', 'startFrame': 1, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-1', 'path': 'Car2/img', 'startFrame': 46, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-2', 'path': 'Car2/img', 'startFrame': 73, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-3', 'path': 'Car2/img', 'startFrame': 219, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-4', 'path': 'Car2/img', 'startFrame': 255, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-5', 'path': 'Car2/img', 'startFrame': 274, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-6', 'path': 'Car2/img', 'startFrame': 219, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-7', 'path': 'Car2/img', 'startFrame': 192, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-8', 'path': 'Car2/img', 'startFrame': 583, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car2-9', 'path': 'Car2/img', 'startFrame': 82, 'endFrame': 913, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car2/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-0', 'path': 'Car24/img', 'startFrame': 1, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-1', 'path': 'Car24/img', 'startFrame': 92, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-2', 'path': 'Car24/img', 'startFrame': 306, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-3', 'path': 'Car24/img', 'startFrame': 275, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-4', 'path': 'Car24/img', 'startFrame': 367, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-5', 'path': 'Car24/img', 'startFrame': 1526, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-6', 'path': 'Car24/img', 'startFrame': 1281, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-7', 'path': 'Car24/img', 'startFrame': 1068, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-8', 'path': 'Car24/img', 'startFrame': 245, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car24-9', 'path': 'Car24/img', 'startFrame': 2746, 'endFrame': 3059, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car24/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-0', 'path': 'Car4/img', 'startFrame': 1, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-1', 'path': 'Car4/img', 'startFrame': 53, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-2', 'path': 'Car4/img', 'startFrame': 14, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-3', 'path': 'Car4/img', 'startFrame': 157, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-4', 'path': 'Car4/img', 'startFrame': 131, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-5', 'path': 'Car4/img', 'startFrame': 98, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-6', 'path': 'Car4/img', 'startFrame': 196, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-7', 'path': 'Car4/img', 'startFrame': 274, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-8', 'path': 'Car4/img', 'startFrame': 261, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Car4-9', 'path': 'Car4/img', 'startFrame': 118, 'endFrame': 659, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Car4/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-0', 'path': 'CarDark/img', 'startFrame': 1, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-1', 'path': 'CarDark/img', 'startFrame': 36, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-2', 'path': 'CarDark/img', 'startFrame': 8, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-3', 'path': 'CarDark/img', 'startFrame': 36, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-4', 'path': 'CarDark/img', 'startFrame': 94, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-5', 'path': 'CarDark/img', 'startFrame': 118, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-6', 'path': 'CarDark/img', 'startFrame': 164, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-7', 'path': 'CarDark/img', 'startFrame': 55, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-8', 'path': 'CarDark/img', 'startFrame': 188, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarDark-9', 'path': 'CarDark/img', 'startFrame': 352, 'endFrame': 393, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarDark/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-0', 'path': 'CarScale/img', 'startFrame': 1, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-1', 'path': 'CarScale/img', 'startFrame': 18, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-2', 'path': 'CarScale/img', 'startFrame': 41, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-3', 'path': 'CarScale/img', 'startFrame': 31, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-4', 'path': 'CarScale/img', 'startFrame': 101, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-5', 'path': 'CarScale/img', 'startFrame': 88, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-6', 'path': 'CarScale/img', 'startFrame': 61, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-7', 'path': 'CarScale/img', 'startFrame': 71, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-8', 'path': 'CarScale/img', 'startFrame': 101, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'CarScale-9', 'path': 'CarScale/img', 'startFrame': 136, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'CarScale/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'ClifBar-0', 'path': 'ClifBar/img', 'startFrame': 1, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-1', 'path': 'ClifBar/img', 'startFrame': 48, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-2', 'path': 'ClifBar/img', 'startFrame': 66, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-3', 'path': 'ClifBar/img', 'startFrame': 43, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-4', 'path': 'ClifBar/img', 'startFrame': 151, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-5', 'path': 'ClifBar/img', 'startFrame': 236, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-6', 'path': 'ClifBar/img', 'startFrame': 198, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-7', 'path': 'ClifBar/img', 'startFrame': 330, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-8', 'path': 'ClifBar/img', 'startFrame': 151, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'ClifBar-9', 'path': 'ClifBar/img', 'startFrame': 212, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'ClifBar/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-0', 'path': 'Coke/img', 'startFrame': 1, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-1', 'path': 'Coke/img', 'startFrame': 3, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-2', 'path': 'Coke/img', 'startFrame': 35, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-3', 'path': 'Coke/img', 'startFrame': 61, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-4', 'path': 'Coke/img', 'startFrame': 12, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-5', 'path': 'Coke/img', 'startFrame': 102, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-6', 'path': 'Coke/img', 'startFrame': 88, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-7', 'path': 'Coke/img', 'startFrame': 163, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-8', 'path': 'Coke/img', 'startFrame': 209, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coke-9', 'path': 'Coke/img', 'startFrame': 131, 'endFrame': 291, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coke/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Couple-0', 'path': 'Couple/img', 'startFrame': 1, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-1', 'path': 'Couple/img', 'startFrame': 9, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-2', 'path': 'Couple/img', 'startFrame': 23, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-3', 'path': 'Couple/img', 'startFrame': 9, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-4', 'path': 'Couple/img', 'startFrame': 40, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-5', 'path': 'Couple/img', 'startFrame': 15, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-6', 'path': 'Couple/img', 'startFrame': 59, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-7', 'path': 'Couple/img', 'startFrame': 10, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-8', 'path': 'Couple/img', 'startFrame': 34, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Couple-9', 'path': 'Couple/img', 'startFrame': 127, 'endFrame': 140, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Couple/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Coupon-0', 'path': 'Coupon/img', 'startFrame': 1, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-1', 'path': 'Coupon/img', 'startFrame': 20, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-2', 'path': 'Coupon/img', 'startFrame': 26, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-3', 'path': 'Coupon/img', 'startFrame': 39, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-4', 'path': 'Coupon/img', 'startFrame': 129, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-5', 'path': 'Coupon/img', 'startFrame': 113, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-6', 'path': 'Coupon/img', 'startFrame': 116, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-7', 'path': 'Coupon/img', 'startFrame': 23, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-8', 'path': 'Coupon/img', 'startFrame': 129, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Coupon-9', 'path': 'Coupon/img', 'startFrame': 289, 'endFrame': 327, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Coupon/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Crossing-0', 'path': 'Crossing/img', 'startFrame': 1, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-1', 'path': 'Crossing/img', 'startFrame': 3, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-2', 'path': 'Crossing/img', 'startFrame': 20, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-3', 'path': 'Crossing/img', 'startFrame': 37, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-4', 'path': 'Crossing/img', 'startFrame': 25, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-5', 'path': 'Crossing/img', 'startFrame': 31, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-6', 'path': 'Crossing/img', 'startFrame': 15, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-7', 'path': 'Crossing/img', 'startFrame': 68, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-8', 'path': 'Crossing/img', 'startFrame': 39, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crossing-9', 'path': 'Crossing/img', 'startFrame': 55, 'endFrame': 120, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crossing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-0', 'path': 'Crowds/img', 'startFrame': 1, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-1', 'path': 'Crowds/img', 'startFrame': 4, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-2', 'path': 'Crowds/img', 'startFrame': 28, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-3', 'path': 'Crowds/img', 'startFrame': 103, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-4', 'path': 'Crowds/img', 'startFrame': 82, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-5', 'path': 'Crowds/img', 'startFrame': 103, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-6', 'path': 'Crowds/img', 'startFrame': 184, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-7', 'path': 'Crowds/img', 'startFrame': 191, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-8', 'path': 'Crowds/img', 'startFrame': 55, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Crowds-9', 'path': 'Crowds/img', 'startFrame': 123, 'endFrame': 347, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Crowds/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-0', 'path': 'Dancer/img', 'startFrame': 1, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-1', 'path': 'Dancer/img', 'startFrame': 5, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-2', 'path': 'Dancer/img', 'startFrame': 31, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-3', 'path': 'Dancer/img', 'startFrame': 27, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-4', 'path': 'Dancer/img', 'startFrame': 18, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-5', 'path': 'Dancer/img', 'startFrame': 45, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-6', 'path': 'Dancer/img', 'startFrame': 133, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-7', 'path': 'Dancer/img', 'startFrame': 62, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-8', 'path': 'Dancer/img', 'startFrame': 36, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer-9', 'path': 'Dancer/img', 'startFrame': 60, 'endFrame': 225, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-0', 'path': 'Dancer2/img', 'startFrame': 1, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-1', 'path': 'Dancer2/img', 'startFrame': 4, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-2', 'path': 'Dancer2/img', 'startFrame': 19, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-3', 'path': 'Dancer2/img', 'startFrame': 5, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-4', 'path': 'Dancer2/img', 'startFrame': 37, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-5', 'path': 'Dancer2/img', 'startFrame': 23, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-6', 'path': 'Dancer2/img', 'startFrame': 46, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-7', 'path': 'Dancer2/img', 'startFrame': 11, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-8', 'path': 'Dancer2/img', 'startFrame': 73, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dancer2-9', 'path': 'Dancer2/img', 'startFrame': 41, 'endFrame': 150, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dancer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David-0', 'path': 'David/img', 'startFrame': 300, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-1', 'path': 'David/img', 'startFrame': 317, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-2', 'path': 'David/img', 'startFrame': 306, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-3', 'path': 'David/img', 'startFrame': 335, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-4', 'path': 'David/img', 'startFrame': 354, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-5', 'path': 'David/img', 'startFrame': 317, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-6', 'path': 'David/img', 'startFrame': 402, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-7', 'path': 'David/img', 'startFrame': 311, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-8', 'path': 'David/img', 'startFrame': 395, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David-9', 'path': 'David/img', 'startFrame': 315, 'endFrame': 770, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-0', 'path': 'David2/img', 'startFrame': 1, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-1', 'path': 'David2/img', 'startFrame': 43, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-2', 'path': 'David2/img', 'startFrame': 11, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-3', 'path': 'David2/img', 'startFrame': 160, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-4', 'path': 'David2/img', 'startFrame': 43, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-5', 'path': 'David2/img', 'startFrame': 107, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-6', 'path': 'David2/img', 'startFrame': 287, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-7', 'path': 'David2/img', 'startFrame': 112, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-8', 'path': 'David2/img', 'startFrame': 170, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David2-9', 'path': 'David2/img', 'startFrame': 478, 'endFrame': 537, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'David3-0', 'path': 'David3/img', 'startFrame': 1, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-1', 'path': 'David3/img', 'startFrame': 26, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-2', 'path': 'David3/img', 'startFrame': 46, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-3', 'path': 'David3/img', 'startFrame': 31, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-4', 'path': 'David3/img', 'startFrame': 41, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-5', 'path': 'David3/img', 'startFrame': 76, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-6', 'path': 'David3/img', 'startFrame': 151, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-7', 'path': 'David3/img', 'startFrame': 53, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-8', 'path': 'David3/img', 'startFrame': 41, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'David3-9', 'path': 'David3/img', 'startFrame': 136, 'endFrame': 252, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'David3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Deer-0', 'path': 'Deer/img', 'startFrame': 1, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-1', 'path': 'Deer/img', 'startFrame': 2, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-2', 'path': 'Deer/img', 'startFrame': 9, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-3', 'path': 'Deer/img', 'startFrame': 7, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-4', 'path': 'Deer/img', 'startFrame': 29, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-5', 'path': 'Deer/img', 'startFrame': 8, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-6', 'path': 'Deer/img', 'startFrame': 17, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-7', 'path': 'Deer/img', 'startFrame': 20, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-8', 'path': 'Deer/img', 'startFrame': 34, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Deer-9', 'path': 'Deer/img', 'startFrame': 7, 'endFrame': 71, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Deer/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Diving-0', 'path': 'Diving/img', 'startFrame': 1, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-1', 'path': 'Diving/img', 'startFrame': 19, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-2', 'path': 'Diving/img', 'startFrame': 43, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-3', 'path': 'Diving/img', 'startFrame': 64, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-4', 'path': 'Diving/img', 'startFrame': 34, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-5', 'path': 'Diving/img', 'startFrame': 11, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-6', 'path': 'Diving/img', 'startFrame': 76, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-7', 'path': 'Diving/img', 'startFrame': 89, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-8', 'path': 'Diving/img', 'startFrame': 152, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Diving-9', 'path': 'Diving/img', 'startFrame': 114, 'endFrame': 215, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Diving/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Dog-0', 'path': 'Dog/img', 'startFrame': 1, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-1', 'path': 'Dog/img', 'startFrame': 5, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-2', 'path': 'Dog/img', 'startFrame': 17, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-3', 'path': 'Dog/img', 'startFrame': 26, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-4', 'path': 'Dog/img', 'startFrame': 5, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-5', 'path': 'Dog/img', 'startFrame': 13, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-6', 'path': 'Dog/img', 'startFrame': 8, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-7', 'path': 'Dog/img', 'startFrame': 26, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-8', 'path': 'Dog/img', 'startFrame': 68, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog-9', 'path': 'Dog/img', 'startFrame': 76, 'endFrame': 127, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-0', 'path': 'Dog1/img', 'startFrame': 1, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-1', 'path': 'Dog1/img', 'startFrame': 136, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-2', 'path': 'Dog1/img', 'startFrame': 217, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-3', 'path': 'Dog1/img', 'startFrame': 284, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-4', 'path': 'Dog1/img', 'startFrame': 217, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-5', 'path': 'Dog1/img', 'startFrame': 406, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-6', 'path': 'Dog1/img', 'startFrame': 406, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-7', 'path': 'Dog1/img', 'startFrame': 284, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-8', 'path': 'Dog1/img', 'startFrame': 325, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Dog1-9', 'path': 'Dog1/img', 'startFrame': 244, 'endFrame': 1350, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dog1/groundtruth_rect.txt', 'object_class': 'dog'}
#             ,
#             {'name': 'Doll-0', 'path': 'Doll/img', 'startFrame': 1, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-1', 'path': 'Doll/img', 'startFrame': 117, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-2', 'path': 'Doll/img', 'startFrame': 78, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-3', 'path': 'Doll/img', 'startFrame': 1045, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-4', 'path': 'Doll/img', 'startFrame': 1084, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-5', 'path': 'Doll/img', 'startFrame': 1936, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-6', 'path': 'Doll/img', 'startFrame': 929, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-7', 'path': 'Doll/img', 'startFrame': 542, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-8', 'path': 'Doll/img', 'startFrame': 929, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Doll-9', 'path': 'Doll/img', 'startFrame': 349, 'endFrame': 3872, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Doll/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'DragonBaby-0', 'path': 'DragonBaby/img', 'startFrame': 1, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-1', 'path': 'DragonBaby/img', 'startFrame': 8, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-2', 'path': 'DragonBaby/img', 'startFrame': 14, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-3', 'path': 'DragonBaby/img', 'startFrame': 7, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-4', 'path': 'DragonBaby/img', 'startFrame': 18, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-5', 'path': 'DragonBaby/img', 'startFrame': 45, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-6', 'path': 'DragonBaby/img', 'startFrame': 34, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-7', 'path': 'DragonBaby/img', 'startFrame': 24, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-8', 'path': 'DragonBaby/img', 'startFrame': 9, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'DragonBaby-9', 'path': 'DragonBaby/img', 'startFrame': 50, 'endFrame': 113, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'DragonBaby/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-0', 'path': 'Dudek/img', 'startFrame': 1, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-1', 'path': 'Dudek/img', 'startFrame': 58, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-2', 'path': 'Dudek/img', 'startFrame': 23, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-3', 'path': 'Dudek/img', 'startFrame': 69, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-4', 'path': 'Dudek/img', 'startFrame': 92, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-5', 'path': 'Dudek/img', 'startFrame': 343, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-6', 'path': 'Dudek/img', 'startFrame': 548, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-7', 'path': 'Dudek/img', 'startFrame': 639, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-8', 'path': 'Dudek/img', 'startFrame': 821, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Dudek-9', 'path': 'Dudek/img', 'startFrame': 924, 'endFrame': 1145, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Dudek/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-0', 'path': 'FaceOcc1/img', 'startFrame': 1, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-1', 'path': 'FaceOcc1/img', 'startFrame': 27, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-2', 'path': 'FaceOcc1/img', 'startFrame': 72, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-3', 'path': 'FaceOcc1/img', 'startFrame': 107, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-4', 'path': 'FaceOcc1/img', 'startFrame': 321, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-5', 'path': 'FaceOcc1/img', 'startFrame': 357, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-6', 'path': 'FaceOcc1/img', 'startFrame': 107, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-7', 'path': 'FaceOcc1/img', 'startFrame': 499, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-8', 'path': 'FaceOcc1/img', 'startFrame': 214, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc1-9', 'path': 'FaceOcc1/img', 'startFrame': 161, 'endFrame': 892, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-0', 'path': 'FaceOcc2/img', 'startFrame': 1, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-1', 'path': 'FaceOcc2/img', 'startFrame': 73, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-2', 'path': 'FaceOcc2/img', 'startFrame': 114, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-3', 'path': 'FaceOcc2/img', 'startFrame': 98, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-4', 'path': 'FaceOcc2/img', 'startFrame': 260, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-5', 'path': 'FaceOcc2/img', 'startFrame': 203, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-6', 'path': 'FaceOcc2/img', 'startFrame': 292, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-7', 'path': 'FaceOcc2/img', 'startFrame': 397, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-8', 'path': 'FaceOcc2/img', 'startFrame': 195, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FaceOcc2-9', 'path': 'FaceOcc2/img', 'startFrame': 73, 'endFrame': 812, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FaceOcc2/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Fish-0', 'path': 'Fish/img', 'startFrame': 1, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-1', 'path': 'Fish/img', 'startFrame': 15, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-2', 'path': 'Fish/img', 'startFrame': 29, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-3', 'path': 'Fish/img', 'startFrame': 29, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-4', 'path': 'Fish/img', 'startFrame': 132, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-5', 'path': 'Fish/img', 'startFrame': 212, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-6', 'path': 'Fish/img', 'startFrame': 85, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-7', 'path': 'Fish/img', 'startFrame': 66, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-8', 'path': 'Fish/img', 'startFrame': 377, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Fish-9', 'path': 'Fish/img', 'startFrame': 254, 'endFrame': 476, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Fish/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'FleetFace-0', 'path': 'FleetFace/img', 'startFrame': 1, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-1', 'path': 'FleetFace/img', 'startFrame': 29, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-2', 'path': 'FleetFace/img', 'startFrame': 127, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-3', 'path': 'FleetFace/img', 'startFrame': 22, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-4', 'path': 'FleetFace/img', 'startFrame': 253, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-5', 'path': 'FleetFace/img', 'startFrame': 176, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-6', 'path': 'FleetFace/img', 'startFrame': 379, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-7', 'path': 'FleetFace/img', 'startFrame': 295, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-8', 'path': 'FleetFace/img', 'startFrame': 113, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'FleetFace-9', 'path': 'FleetFace/img', 'startFrame': 631, 'endFrame': 707, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'FleetFace/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football-0', 'path': 'Football/img', 'startFrame': 1, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-1', 'path': 'Football/img', 'startFrame': 8, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-2', 'path': 'Football/img', 'startFrame': 73, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-3', 'path': 'Football/img', 'startFrame': 44, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-4', 'path': 'Football/img', 'startFrame': 73, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-5', 'path': 'Football/img', 'startFrame': 181, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-6', 'path': 'Football/img', 'startFrame': 87, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-7', 'path': 'Football/img', 'startFrame': 177, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-8', 'path': 'Football/img', 'startFrame': 202, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football-9', 'path': 'Football/img', 'startFrame': 98, 'endFrame': 362, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Football1-0', 'path': 'Football1/img', 'startFrame': 1, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-1', 'path': 'Football1/img', 'startFrame': 8, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-2', 'path': 'Football1/img', 'startFrame': 9, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-3', 'path': 'Football1/img', 'startFrame': 15, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-4', 'path': 'Football1/img', 'startFrame': 12, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-5', 'path': 'Football1/img', 'startFrame': 25, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-6', 'path': 'Football1/img', 'startFrame': 22, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-7', 'path': 'Football1/img', 'startFrame': 25, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-8', 'path': 'Football1/img', 'startFrame': 57, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Football1-9', 'path': 'Football1/img', 'startFrame': 13, 'endFrame': 74, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Football1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-0', 'path': 'Freeman1/img', 'startFrame': 1, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-1', 'path': 'Freeman1/img', 'startFrame': 29, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-2', 'path': 'Freeman1/img', 'startFrame': 13, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-3', 'path': 'Freeman1/img', 'startFrame': 97, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-4', 'path': 'Freeman1/img', 'startFrame': 13, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-5', 'path': 'Freeman1/img', 'startFrame': 65, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-6', 'path': 'Freeman1/img', 'startFrame': 193, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-7', 'path': 'Freeman1/img', 'startFrame': 157, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-8', 'path': 'Freeman1/img', 'startFrame': 77, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman1-9', 'path': 'Freeman1/img', 'startFrame': 260, 'endFrame': 326, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman1/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-0', 'path': 'Freeman3/img', 'startFrame': 1, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-1', 'path': 'Freeman3/img', 'startFrame': 10, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-2', 'path': 'Freeman3/img', 'startFrame': 19, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-3', 'path': 'Freeman3/img', 'startFrame': 111, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-4', 'path': 'Freeman3/img', 'startFrame': 74, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-5', 'path': 'Freeman3/img', 'startFrame': 116, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-6', 'path': 'Freeman3/img', 'startFrame': 166, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-7', 'path': 'Freeman3/img', 'startFrame': 129, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-8', 'path': 'Freeman3/img', 'startFrame': 369, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman3-9', 'path': 'Freeman3/img', 'startFrame': 415, 'endFrame': 460, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman3/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-0', 'path': 'Freeman4/img', 'startFrame': 1, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-1', 'path': 'Freeman4/img', 'startFrame': 23, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-2', 'path': 'Freeman4/img', 'startFrame': 12, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-3', 'path': 'Freeman4/img', 'startFrame': 9, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-4', 'path': 'Freeman4/img', 'startFrame': 23, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-5', 'path': 'Freeman4/img', 'startFrame': 127, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-6', 'path': 'Freeman4/img', 'startFrame': 101, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-7', 'path': 'Freeman4/img', 'startFrame': 99, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-8', 'path': 'Freeman4/img', 'startFrame': 113, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Freeman4-9', 'path': 'Freeman4/img', 'startFrame': 227, 'endFrame': 283, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Freeman4/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-0', 'path': 'Girl/img', 'startFrame': 1, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-1', 'path': 'Girl/img', 'startFrame': 11, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-2', 'path': 'Girl/img', 'startFrame': 11, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-3', 'path': 'Girl/img', 'startFrame': 121, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-4', 'path': 'Girl/img', 'startFrame': 181, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-5', 'path': 'Girl/img', 'startFrame': 126, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-6', 'path': 'Girl/img', 'startFrame': 90, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-7', 'path': 'Girl/img', 'startFrame': 141, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-8', 'path': 'Girl/img', 'startFrame': 241, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl-9', 'path': 'Girl/img', 'startFrame': 181, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Girl2-0', 'path': 'Girl2/img', 'startFrame': 1, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-1', 'path': 'Girl2/img', 'startFrame': 121, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-2', 'path': 'Girl2/img', 'startFrame': 181, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-3', 'path': 'Girl2/img', 'startFrame': 226, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-4', 'path': 'Girl2/img', 'startFrame': 601, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-5', 'path': 'Girl2/img', 'startFrame': 676, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-6', 'path': 'Girl2/img', 'startFrame': 811, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-7', 'path': 'Girl2/img', 'startFrame': 735, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-8', 'path': 'Girl2/img', 'startFrame': 721, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Girl2-9', 'path': 'Girl2/img', 'startFrame': 810, 'endFrame': 1500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Girl2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-0', 'path': 'Gym/img', 'startFrame': 1, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-1', 'path': 'Gym/img', 'startFrame': 77, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-2', 'path': 'Gym/img', 'startFrame': 46, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-3', 'path': 'Gym/img', 'startFrame': 115, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-4', 'path': 'Gym/img', 'startFrame': 183, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-5', 'path': 'Gym/img', 'startFrame': 191, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-6', 'path': 'Gym/img', 'startFrame': 457, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-7', 'path': 'Gym/img', 'startFrame': 213, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-8', 'path': 'Gym/img', 'startFrame': 244, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Gym-9', 'path': 'Gym/img', 'startFrame': 206, 'endFrame': 767, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Gym/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-0', 'path': 'Human2/img', 'startFrame': 1, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-1', 'path': 'Human2/img', 'startFrame': 113, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-2', 'path': 'Human2/img', 'startFrame': 113, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-3', 'path': 'Human2/img', 'startFrame': 337, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-4', 'path': 'Human2/img', 'startFrame': 269, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-5', 'path': 'Human2/img', 'startFrame': 561, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-6', 'path': 'Human2/img', 'startFrame': 68, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-7', 'path': 'Human2/img', 'startFrame': 628, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-8', 'path': 'Human2/img', 'startFrame': 449, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human2-9', 'path': 'Human2/img', 'startFrame': 303, 'endFrame': 1128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-0', 'path': 'Human3/img', 'startFrame': 1, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-1', 'path': 'Human3/img', 'startFrame': 68, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-2', 'path': 'Human3/img', 'startFrame': 170, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-3', 'path': 'Human3/img', 'startFrame': 102, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-4', 'path': 'Human3/img', 'startFrame': 339, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-5', 'path': 'Human3/img', 'startFrame': 846, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-6', 'path': 'Human3/img', 'startFrame': 609, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-7', 'path': 'Human3/img', 'startFrame': 474, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-8', 'path': 'Human3/img', 'startFrame': 1217, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human3-9', 'path': 'Human3/img', 'startFrame': 1217, 'endFrame': 1698, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human3/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-0', 'path': 'Human4/img', 'startFrame': 1, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-1', 'path': 'Human4/img', 'startFrame': 67, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-2', 'path': 'Human4/img', 'startFrame': 53, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-3', 'path': 'Human4/img', 'startFrame': 119, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-4', 'path': 'Human4/img', 'startFrame': 265, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-5', 'path': 'Human4/img', 'startFrame': 166, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-6', 'path': 'Human4/img', 'startFrame': 317, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-7', 'path': 'Human4/img', 'startFrame': 232, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-8', 'path': 'Human4/img', 'startFrame': 529, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human4_2-9', 'path': 'Human4/img', 'startFrame': 119, 'endFrame': 667, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human4/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-0', 'path': 'Human5/img', 'startFrame': 1, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-1', 'path': 'Human5/img', 'startFrame': 50, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-2', 'path': 'Human5/img', 'startFrame': 143, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-3', 'path': 'Human5/img', 'startFrame': 150, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-4', 'path': 'Human5/img', 'startFrame': 171, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-5', 'path': 'Human5/img', 'startFrame': 320, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-6', 'path': 'Human5/img', 'startFrame': 171, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-7', 'path': 'Human5/img', 'startFrame': 100, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-8', 'path': 'Human5/img', 'startFrame': 455, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human5-9', 'path': 'Human5/img', 'startFrame': 512, 'endFrame': 713, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human5/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-0', 'path': 'Human6/img', 'startFrame': 1, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-1', 'path': 'Human6/img', 'startFrame': 24, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-2', 'path': 'Human6/img', 'startFrame': 16, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-3', 'path': 'Human6/img', 'startFrame': 24, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-4', 'path': 'Human6/img', 'startFrame': 285, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-5', 'path': 'Human6/img', 'startFrame': 80, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-6', 'path': 'Human6/img', 'startFrame': 190, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-7', 'path': 'Human6/img', 'startFrame': 554, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-8', 'path': 'Human6/img', 'startFrame': 127, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human6-9', 'path': 'Human6/img', 'startFrame': 72, 'endFrame': 792, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human6/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-0', 'path': 'Human7/img', 'startFrame': 1, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-1', 'path': 'Human7/img', 'startFrame': 23, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-2', 'path': 'Human7/img', 'startFrame': 51, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-3', 'path': 'Human7/img', 'startFrame': 45, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-4', 'path': 'Human7/img', 'startFrame': 11, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-5', 'path': 'Human7/img', 'startFrame': 113, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-6', 'path': 'Human7/img', 'startFrame': 61, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-7', 'path': 'Human7/img', 'startFrame': 53, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-8', 'path': 'Human7/img', 'startFrame': 101, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human7-9', 'path': 'Human7/img', 'startFrame': 46, 'endFrame': 250, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human7/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-0', 'path': 'Human8/img', 'startFrame': 1, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-1', 'path': 'Human8/img', 'startFrame': 13, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-2', 'path': 'Human8/img', 'startFrame': 17, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-3', 'path': 'Human8/img', 'startFrame': 15, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-4', 'path': 'Human8/img', 'startFrame': 5, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-5', 'path': 'Human8/img', 'startFrame': 31, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-6', 'path': 'Human8/img', 'startFrame': 73, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-7', 'path': 'Human8/img', 'startFrame': 17, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-8', 'path': 'Human8/img', 'startFrame': 68, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human8-9', 'path': 'Human8/img', 'startFrame': 87, 'endFrame': 128, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human8/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-0', 'path': 'Human9/img', 'startFrame': 1, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-1', 'path': 'Human9/img', 'startFrame': 28, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-2', 'path': 'Human9/img', 'startFrame': 7, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-3', 'path': 'Human9/img', 'startFrame': 73, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-4', 'path': 'Human9/img', 'startFrame': 13, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-5', 'path': 'Human9/img', 'startFrame': 76, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-6', 'path': 'Human9/img', 'startFrame': 108, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-7', 'path': 'Human9/img', 'startFrame': 147, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-8', 'path': 'Human9/img', 'startFrame': 145, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Human9-9', 'path': 'Human9/img', 'startFrame': 162, 'endFrame': 305, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Human9/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Ironman-0', 'path': 'Ironman/img', 'startFrame': 1, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-1', 'path': 'Ironman/img', 'startFrame': 7, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-2', 'path': 'Ironman/img', 'startFrame': 20, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-3', 'path': 'Ironman/img', 'startFrame': 25, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-4', 'path': 'Ironman/img', 'startFrame': 52, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-5', 'path': 'Ironman/img', 'startFrame': 41, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-6', 'path': 'Ironman/img', 'startFrame': 10, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-7', 'path': 'Ironman/img', 'startFrame': 45, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-8', 'path': 'Ironman/img', 'startFrame': 103, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Ironman-9', 'path': 'Ironman/img', 'startFrame': 87, 'endFrame': 166, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Ironman/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Jogging_1-0', 'path': 'Jogging/img', 'startFrame': 1, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-1', 'path': 'Jogging/img', 'startFrame': 28, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-2', 'path': 'Jogging/img', 'startFrame': 37, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-3', 'path': 'Jogging/img', 'startFrame': 37, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-4', 'path': 'Jogging/img', 'startFrame': 109, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-5', 'path': 'Jogging/img', 'startFrame': 91, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-6', 'path': 'Jogging/img', 'startFrame': 73, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-7', 'path': 'Jogging/img', 'startFrame': 64, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-8', 'path': 'Jogging/img', 'startFrame': 97, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_1-9', 'path': 'Jogging/img', 'startFrame': 162, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-0', 'path': 'Jogging/img', 'startFrame': 1, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-1', 'path': 'Jogging/img', 'startFrame': 7, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-2', 'path': 'Jogging/img', 'startFrame': 31, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-3', 'path': 'Jogging/img', 'startFrame': 63, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-4', 'path': 'Jogging/img', 'startFrame': 109, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-5', 'path': 'Jogging/img', 'startFrame': 46, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-6', 'path': 'Jogging/img', 'startFrame': 73, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-7', 'path': 'Jogging/img', 'startFrame': 169, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-8', 'path': 'Jogging/img', 'startFrame': 49, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jogging_2-9', 'path': 'Jogging/img', 'startFrame': 162, 'endFrame': 307, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jogging/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-0', 'path': 'Jump/img', 'startFrame': 1, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-1', 'path': 'Jump/img', 'startFrame': 8, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-2', 'path': 'Jump/img', 'startFrame': 8, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-3', 'path': 'Jump/img', 'startFrame': 15, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-4', 'path': 'Jump/img', 'startFrame': 39, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-5', 'path': 'Jump/img', 'startFrame': 61, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-6', 'path': 'Jump/img', 'startFrame': 37, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-7', 'path': 'Jump/img', 'startFrame': 76, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-8', 'path': 'Jump/img', 'startFrame': 29, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jump-9', 'path': 'Jump/img', 'startFrame': 65, 'endFrame': 122, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jump/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Jumping-0', 'path': 'Jumping/img', 'startFrame': 1, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-1', 'path': 'Jumping/img', 'startFrame': 16, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-2', 'path': 'Jumping/img', 'startFrame': 13, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-3', 'path': 'Jumping/img', 'startFrame': 56, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-4', 'path': 'Jumping/img', 'startFrame': 112, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-5', 'path': 'Jumping/img', 'startFrame': 16, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-6', 'path': 'Jumping/img', 'startFrame': 75, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-7', 'path': 'Jumping/img', 'startFrame': 87, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-8', 'path': 'Jumping/img', 'startFrame': 249, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Jumping-9', 'path': 'Jumping/img', 'startFrame': 84, 'endFrame': 313, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Jumping/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-0', 'path': 'KiteSurf/img', 'startFrame': 1, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-1', 'path': 'KiteSurf/img', 'startFrame': 9, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-2', 'path': 'KiteSurf/img', 'startFrame': 12, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-3', 'path': 'KiteSurf/img', 'startFrame': 3, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-4', 'path': 'KiteSurf/img', 'startFrame': 26, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-5', 'path': 'KiteSurf/img', 'startFrame': 37, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-6', 'path': 'KiteSurf/img', 'startFrame': 49, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-7', 'path': 'KiteSurf/img', 'startFrame': 34, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-8', 'path': 'KiteSurf/img', 'startFrame': 20, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'KiteSurf-9', 'path': 'KiteSurf/img', 'startFrame': 22, 'endFrame': 84, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'KiteSurf/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Lemming-0', 'path': 'Lemming/img', 'startFrame': 1, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-1', 'path': 'Lemming/img', 'startFrame': 80, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-2', 'path': 'Lemming/img', 'startFrame': 240, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-3', 'path': 'Lemming/img', 'startFrame': 160, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-4', 'path': 'Lemming/img', 'startFrame': 426, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-5', 'path': 'Lemming/img', 'startFrame': 666, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-6', 'path': 'Lemming/img', 'startFrame': 719, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-7', 'path': 'Lemming/img', 'startFrame': 559, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-8', 'path': 'Lemming/img', 'startFrame': 639, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Lemming-9', 'path': 'Lemming/img', 'startFrame': 599, 'endFrame': 1336, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Lemming/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-0', 'path': 'Liquor/img', 'startFrame': 1, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-1', 'path': 'Liquor/img', 'startFrame': 175, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-2', 'path': 'Liquor/img', 'startFrame': 70, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-3', 'path': 'Liquor/img', 'startFrame': 314, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-4', 'path': 'Liquor/img', 'startFrame': 418, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-5', 'path': 'Liquor/img', 'startFrame': 784, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-6', 'path': 'Liquor/img', 'startFrame': 418, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-7', 'path': 'Liquor/img', 'startFrame': 1097, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-8', 'path': 'Liquor/img', 'startFrame': 557, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Liquor-9', 'path': 'Liquor/img', 'startFrame': 627, 'endFrame': 1741, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Liquor/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Man-0', 'path': 'Man/img', 'startFrame': 1, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-1', 'path': 'Man/img', 'startFrame': 7, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-2', 'path': 'Man/img', 'startFrame': 21, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-3', 'path': 'Man/img', 'startFrame': 32, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-4', 'path': 'Man/img', 'startFrame': 42, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-5', 'path': 'Man/img', 'startFrame': 14, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-6', 'path': 'Man/img', 'startFrame': 24, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-7', 'path': 'Man/img', 'startFrame': 37, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-8', 'path': 'Man/img', 'startFrame': 11, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Man-9', 'path': 'Man/img', 'startFrame': 12, 'endFrame': 134, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Man/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Matrix-0', 'path': 'Matrix/img', 'startFrame': 1, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-1', 'path': 'Matrix/img', 'startFrame': 3, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-2', 'path': 'Matrix/img', 'startFrame': 11, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-3', 'path': 'Matrix/img', 'startFrame': 25, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-4', 'path': 'Matrix/img', 'startFrame': 21, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-5', 'path': 'Matrix/img', 'startFrame': 41, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-6', 'path': 'Matrix/img', 'startFrame': 13, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-7', 'path': 'Matrix/img', 'startFrame': 36, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-8', 'path': 'Matrix/img', 'startFrame': 41, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Matrix-9', 'path': 'Matrix/img', 'startFrame': 64, 'endFrame': 100, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Matrix/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Mhyang-0', 'path': 'Mhyang/img', 'startFrame': 1, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-1', 'path': 'Mhyang/img', 'startFrame': 45, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-2', 'path': 'Mhyang/img', 'startFrame': 269, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-3', 'path': 'Mhyang/img', 'startFrame': 313, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-4', 'path': 'Mhyang/img', 'startFrame': 477, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-5', 'path': 'Mhyang/img', 'startFrame': 597, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-6', 'path': 'Mhyang/img', 'startFrame': 805, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-7', 'path': 'Mhyang/img', 'startFrame': 522, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-8', 'path': 'Mhyang/img', 'startFrame': 1073, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Mhyang-9', 'path': 'Mhyang/img', 'startFrame': 671, 'endFrame': 1490, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Mhyang/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'MotorRolling-0', 'path': 'MotorRolling/img', 'startFrame': 1, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-1', 'path': 'MotorRolling/img', 'startFrame': 7, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-2', 'path': 'MotorRolling/img', 'startFrame': 33, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-3', 'path': 'MotorRolling/img', 'startFrame': 10, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-4', 'path': 'MotorRolling/img', 'startFrame': 45, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-5', 'path': 'MotorRolling/img', 'startFrame': 25, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-6', 'path': 'MotorRolling/img', 'startFrame': 77, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-7', 'path': 'MotorRolling/img', 'startFrame': 45, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-8', 'path': 'MotorRolling/img', 'startFrame': 13, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MotorRolling-9', 'path': 'MotorRolling/img', 'startFrame': 15, 'endFrame': 164, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MotorRolling/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'MountainBike-0', 'path': 'MountainBike/img', 'startFrame': 1, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-1', 'path': 'MountainBike/img', 'startFrame': 20, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-2', 'path': 'MountainBike/img', 'startFrame': 5, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-3', 'path': 'MountainBike/img', 'startFrame': 47, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-4', 'path': 'MountainBike/img', 'startFrame': 27, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-5', 'path': 'MountainBike/img', 'startFrame': 45, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-6', 'path': 'MountainBike/img', 'startFrame': 80, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-7', 'path': 'MountainBike/img', 'startFrame': 155, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-8', 'path': 'MountainBike/img', 'startFrame': 159, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'MountainBike-9', 'path': 'MountainBike/img', 'startFrame': 80, 'endFrame': 228, 'nz': 4,
#              'ext': 'jpg', 'anno_path': 'MountainBike/groundtruth_rect.txt', 'object_class': 'bicycle'}
#             ,
#             {'name': 'Panda-0', 'path': 'Panda/img', 'startFrame': 1, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-1', 'path': 'Panda/img', 'startFrame': 91, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-2', 'path': 'Panda/img', 'startFrame': 121, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-3', 'path': 'Panda/img', 'startFrame': 151, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-4', 'path': 'Panda/img', 'startFrame': 401, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-5', 'path': 'Panda/img', 'startFrame': 351, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-6', 'path': 'Panda/img', 'startFrame': 241, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-7', 'path': 'Panda/img', 'startFrame': 421, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-8', 'path': 'Panda/img', 'startFrame': 81, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'Panda-9', 'path': 'Panda/img', 'startFrame': 721, 'endFrame': 1000, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Panda/groundtruth_rect.txt', 'object_class': 'mammal'}
#             ,
#             {'name': 'RedTeam-0', 'path': 'RedTeam/img', 'startFrame': 1, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-1', 'path': 'RedTeam/img', 'startFrame': 96, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-2', 'path': 'RedTeam/img', 'startFrame': 306, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-3', 'path': 'RedTeam/img', 'startFrame': 230, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-4', 'path': 'RedTeam/img', 'startFrame': 230, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-5', 'path': 'RedTeam/img', 'startFrame': 478, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-6', 'path': 'RedTeam/img', 'startFrame': 230, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-7', 'path': 'RedTeam/img', 'startFrame': 268, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-8', 'path': 'RedTeam/img', 'startFrame': 1070, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'RedTeam-9', 'path': 'RedTeam/img', 'startFrame': 860, 'endFrame': 1918, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'RedTeam/groundtruth_rect.txt', 'object_class': 'vehicle'}
#             ,
#             {'name': 'Rubik-0', 'path': 'Rubik/img', 'startFrame': 1, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-1', 'path': 'Rubik/img', 'startFrame': 140, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-2', 'path': 'Rubik/img', 'startFrame': 120, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-3', 'path': 'Rubik/img', 'startFrame': 60, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-4', 'path': 'Rubik/img', 'startFrame': 637, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-5', 'path': 'Rubik/img', 'startFrame': 996, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-6', 'path': 'Rubik/img', 'startFrame': 1195, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-7', 'path': 'Rubik/img', 'startFrame': 976, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-8', 'path': 'Rubik/img', 'startFrame': 1593, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Rubik-9', 'path': 'Rubik/img', 'startFrame': 538, 'endFrame': 1997, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Rubik/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Shaking-0', 'path': 'Shaking/img', 'startFrame': 1, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-1', 'path': 'Shaking/img', 'startFrame': 22, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-2', 'path': 'Shaking/img', 'startFrame': 51, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-3', 'path': 'Shaking/img', 'startFrame': 22, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-4', 'path': 'Shaking/img', 'startFrame': 130, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-5', 'path': 'Shaking/img', 'startFrame': 55, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-6', 'path': 'Shaking/img', 'startFrame': 152, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-7', 'path': 'Shaking/img', 'startFrame': 26, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-8', 'path': 'Shaking/img', 'startFrame': 173, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Shaking-9', 'path': 'Shaking/img', 'startFrame': 227, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Shaking/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Singer1-0', 'path': 'Singer1/img', 'startFrame': 1, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-1', 'path': 'Singer1/img', 'startFrame': 11, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-2', 'path': 'Singer1/img', 'startFrame': 57, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-3', 'path': 'Singer1/img', 'startFrame': 32, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-4', 'path': 'Singer1/img', 'startFrame': 113, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-5', 'path': 'Singer1/img', 'startFrame': 88, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-6', 'path': 'Singer1/img', 'startFrame': 147, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-7', 'path': 'Singer1/img', 'startFrame': 197, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-8', 'path': 'Singer1/img', 'startFrame': 281, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer1-9', 'path': 'Singer1/img', 'startFrame': 64, 'endFrame': 351, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-0', 'path': 'Singer2/img', 'startFrame': 1, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-1', 'path': 'Singer2/img', 'startFrame': 37, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-2', 'path': 'Singer2/img', 'startFrame': 65, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-3', 'path': 'Singer2/img', 'startFrame': 55, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-4', 'path': 'Singer2/img', 'startFrame': 130, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-5', 'path': 'Singer2/img', 'startFrame': 73, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-6', 'path': 'Singer2/img', 'startFrame': 109, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-7', 'path': 'Singer2/img', 'startFrame': 26, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-8', 'path': 'Singer2/img', 'startFrame': 29, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Singer2-9', 'path': 'Singer2/img', 'startFrame': 195, 'endFrame': 366, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Singer2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-0', 'path': 'Skater/img', 'startFrame': 1, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-1', 'path': 'Skater/img', 'startFrame': 17, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-2', 'path': 'Skater/img', 'startFrame': 13, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-3', 'path': 'Skater/img', 'startFrame': 25, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-4', 'path': 'Skater/img', 'startFrame': 20, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-5', 'path': 'Skater/img', 'startFrame': 81, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-6', 'path': 'Skater/img', 'startFrame': 58, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-7', 'path': 'Skater/img', 'startFrame': 101, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-8', 'path': 'Skater/img', 'startFrame': 77, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater-9', 'path': 'Skater/img', 'startFrame': 101, 'endFrame': 160, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-0', 'path': 'Skater2/img', 'startFrame': 1, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-1', 'path': 'Skater2/img', 'startFrame': 26, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-2', 'path': 'Skater2/img', 'startFrame': 9, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-3', 'path': 'Skater2/img', 'startFrame': 65, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-4', 'path': 'Skater2/img', 'startFrame': 173, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-5', 'path': 'Skater2/img', 'startFrame': 108, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-6', 'path': 'Skater2/img', 'startFrame': 26, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-7', 'path': 'Skater2/img', 'startFrame': 302, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-8', 'path': 'Skater2/img', 'startFrame': 173, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skater2-9', 'path': 'Skater2/img', 'startFrame': 78, 'endFrame': 435, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skater2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-0', 'path': 'Skating1/img', 'startFrame': 1, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-1', 'path': 'Skating1/img', 'startFrame': 37, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-2', 'path': 'Skating1/img', 'startFrame': 49, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-3', 'path': 'Skating1/img', 'startFrame': 109, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-4', 'path': 'Skating1/img', 'startFrame': 81, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-5', 'path': 'Skating1/img', 'startFrame': 141, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-6', 'path': 'Skating1/img', 'startFrame': 145, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-7', 'path': 'Skating1/img', 'startFrame': 169, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-8', 'path': 'Skating1/img', 'startFrame': 289, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating1-9', 'path': 'Skating1/img', 'startFrame': 361, 'endFrame': 400, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating1/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-0', 'path': 'Skating2/img', 'startFrame': 1, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-1', 'path': 'Skating2/img', 'startFrame': 15, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-2', 'path': 'Skating2/img', 'startFrame': 85, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-3', 'path': 'Skating2/img', 'startFrame': 15, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-4', 'path': 'Skating2/img', 'startFrame': 132, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-5', 'path': 'Skating2/img', 'startFrame': 142, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-6', 'path': 'Skating2/img', 'startFrame': 283, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-7', 'path': 'Skating2/img', 'startFrame': 330, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-8', 'path': 'Skating2/img', 'startFrame': 301, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_1-9', 'path': 'Skating2/img', 'startFrame': 254, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.1.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-0', 'path': 'Skating2/img', 'startFrame': 1, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-1', 'path': 'Skating2/img', 'startFrame': 19, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-2', 'path': 'Skating2/img', 'startFrame': 38, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-3', 'path': 'Skating2/img', 'startFrame': 85, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-4', 'path': 'Skating2/img', 'startFrame': 95, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-5', 'path': 'Skating2/img', 'startFrame': 95, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-6', 'path': 'Skating2/img', 'startFrame': 198, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-7', 'path': 'Skating2/img', 'startFrame': 330, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-8', 'path': 'Skating2/img', 'startFrame': 151, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skating2_2-9', 'path': 'Skating2/img', 'startFrame': 424, 'endFrame': 473, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skating2/groundtruth_rect.2.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-0', 'path': 'Skiing/img', 'startFrame': 1, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-1', 'path': 'Skiing/img', 'startFrame': 5, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-2', 'path': 'Skiing/img', 'startFrame': 9, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-3', 'path': 'Skiing/img', 'startFrame': 15, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-4', 'path': 'Skiing/img', 'startFrame': 13, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-5', 'path': 'Skiing/img', 'startFrame': 41, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-6', 'path': 'Skiing/img', 'startFrame': 10, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-7', 'path': 'Skiing/img', 'startFrame': 6, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-8', 'path': 'Skiing/img', 'startFrame': 20, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Skiing-9', 'path': 'Skiing/img', 'startFrame': 37, 'endFrame': 81, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Skiing/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Soccer-0', 'path': 'Soccer/img', 'startFrame': 1, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-1', 'path': 'Soccer/img', 'startFrame': 20, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-2', 'path': 'Soccer/img', 'startFrame': 63, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-3', 'path': 'Soccer/img', 'startFrame': 12, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-4', 'path': 'Soccer/img', 'startFrame': 63, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-5', 'path': 'Soccer/img', 'startFrame': 157, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-6', 'path': 'Soccer/img', 'startFrame': 71, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-7', 'path': 'Soccer/img', 'startFrame': 55, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-8', 'path': 'Soccer/img', 'startFrame': 313, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Soccer-9', 'path': 'Soccer/img', 'startFrame': 141, 'endFrame': 392, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Soccer/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Subway-0', 'path': 'Subway/img', 'startFrame': 1, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-1', 'path': 'Subway/img', 'startFrame': 16, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-2', 'path': 'Subway/img', 'startFrame': 28, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-3', 'path': 'Subway/img', 'startFrame': 16, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-4', 'path': 'Subway/img', 'startFrame': 14, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-5', 'path': 'Subway/img', 'startFrame': 9, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-6', 'path': 'Subway/img', 'startFrame': 31, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-7', 'path': 'Subway/img', 'startFrame': 48, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-8', 'path': 'Subway/img', 'startFrame': 55, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Subway-9', 'path': 'Subway/img', 'startFrame': 77, 'endFrame': 175, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Subway/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Surfer-0', 'path': 'Surfer/img', 'startFrame': 1, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-1', 'path': 'Surfer/img', 'startFrame': 8, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-2', 'path': 'Surfer/img', 'startFrame': 52, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-3', 'path': 'Surfer/img', 'startFrame': 78, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-4', 'path': 'Surfer/img', 'startFrame': 134, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-5', 'path': 'Surfer/img', 'startFrame': 75, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-6', 'path': 'Surfer/img', 'startFrame': 134, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-7', 'path': 'Surfer/img', 'startFrame': 208, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-8', 'path': 'Surfer/img', 'startFrame': 30, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Surfer-9', 'path': 'Surfer/img', 'startFrame': 267, 'endFrame': 376, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Surfer/groundtruth_rect.txt', 'object_class': 'person head'}
#             ,
#             {'name': 'Suv-0', 'path': 'Suv/img', 'startFrame': 1, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-1', 'path': 'Suv/img', 'startFrame': 10, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-2', 'path': 'Suv/img', 'startFrame': 151, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-3', 'path': 'Suv/img', 'startFrame': 198, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-4', 'path': 'Suv/img', 'startFrame': 76, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-5', 'path': 'Suv/img', 'startFrame': 330, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-6', 'path': 'Suv/img', 'startFrame': 508, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-7', 'path': 'Suv/img', 'startFrame': 198, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-8', 'path': 'Suv/img', 'startFrame': 301, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Suv-9', 'path': 'Suv/img', 'startFrame': 170, 'endFrame': 945, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Suv/groundtruth_rect.txt', 'object_class': 'car'}
#             ,
#             {'name': 'Sylvester-0', 'path': 'Sylvester/img', 'startFrame': 1, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-1', 'path': 'Sylvester/img', 'startFrame': 14, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-2', 'path': 'Sylvester/img', 'startFrame': 188, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-3', 'path': 'Sylvester/img', 'startFrame': 362, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-4', 'path': 'Sylvester/img', 'startFrame': 54, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-5', 'path': 'Sylvester/img', 'startFrame': 68, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-6', 'path': 'Sylvester/img', 'startFrame': 403, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-7', 'path': 'Sylvester/img', 'startFrame': 845, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-8', 'path': 'Sylvester/img', 'startFrame': 751, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Sylvester-9', 'path': 'Sylvester/img', 'startFrame': 845, 'endFrame': 1345, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Sylvester/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-0', 'path': 'Tiger1/img', 'startFrame': 1, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-1', 'path': 'Tiger1/img', 'startFrame': 29, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-2', 'path': 'Tiger1/img', 'startFrame': 15, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-3', 'path': 'Tiger1/img', 'startFrame': 74, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-4', 'path': 'Tiger1/img', 'startFrame': 85, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-5', 'path': 'Tiger1/img', 'startFrame': 18, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-6', 'path': 'Tiger1/img', 'startFrame': 126, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-7', 'path': 'Tiger1/img', 'startFrame': 25, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-8', 'path': 'Tiger1/img', 'startFrame': 57, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger1-9', 'path': 'Tiger1/img', 'startFrame': 253, 'endFrame': 354, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger1/groundtruth_rect.txt', 'initOmit': 5, 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-0', 'path': 'Tiger2/img', 'startFrame': 1, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-1', 'path': 'Tiger2/img', 'startFrame': 37, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-2', 'path': 'Tiger2/img', 'startFrame': 29, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-3', 'path': 'Tiger2/img', 'startFrame': 109, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-4', 'path': 'Tiger2/img', 'startFrame': 44, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-5', 'path': 'Tiger2/img', 'startFrame': 181, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-6', 'path': 'Tiger2/img', 'startFrame': 217, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-7', 'path': 'Tiger2/img', 'startFrame': 76, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-8', 'path': 'Tiger2/img', 'startFrame': 145, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Tiger2-9', 'path': 'Tiger2/img', 'startFrame': 130, 'endFrame': 365, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Tiger2/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-0', 'path': 'Toy/img', 'startFrame': 1, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-1', 'path': 'Toy/img', 'startFrame': 22, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-2', 'path': 'Toy/img', 'startFrame': 17, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-3', 'path': 'Toy/img', 'startFrame': 65, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-4', 'path': 'Toy/img', 'startFrame': 44, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-5', 'path': 'Toy/img', 'startFrame': 41, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-6', 'path': 'Toy/img', 'startFrame': 17, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-7', 'path': 'Toy/img', 'startFrame': 95, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-8', 'path': 'Toy/img', 'startFrame': 65, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Toy-9', 'path': 'Toy/img', 'startFrame': 219, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Toy/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-0', 'path': 'Trans/img', 'startFrame': 1, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-1', 'path': 'Trans/img', 'startFrame': 7, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-2', 'path': 'Trans/img', 'startFrame': 13, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-3', 'path': 'Trans/img', 'startFrame': 4, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-4', 'path': 'Trans/img', 'startFrame': 5, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-5', 'path': 'Trans/img', 'startFrame': 61, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-6', 'path': 'Trans/img', 'startFrame': 37, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-7', 'path': 'Trans/img', 'startFrame': 26, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-8', 'path': 'Trans/img', 'startFrame': 29, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trans-9', 'path': 'Trans/img', 'startFrame': 109, 'endFrame': 124, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trans/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Trellis-0', 'path': 'Trellis/img', 'startFrame': 1, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-1', 'path': 'Trellis/img', 'startFrame': 34, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-2', 'path': 'Trellis/img', 'startFrame': 113, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-3', 'path': 'Trellis/img', 'startFrame': 135, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-4', 'path': 'Trellis/img', 'startFrame': 225, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-5', 'path': 'Trellis/img', 'startFrame': 281, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-6', 'path': 'Trellis/img', 'startFrame': 135, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-7', 'path': 'Trellis/img', 'startFrame': 236, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-8', 'path': 'Trellis/img', 'startFrame': 135, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Trellis-9', 'path': 'Trellis/img', 'startFrame': 152, 'endFrame': 569, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Trellis/groundtruth_rect.txt', 'object_class': 'face'}
#             ,
#             {'name': 'Twinnings-0', 'path': 'Twinnings/img', 'startFrame': 1, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-1', 'path': 'Twinnings/img', 'startFrame': 38, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-2', 'path': 'Twinnings/img', 'startFrame': 95, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-3', 'path': 'Twinnings/img', 'startFrame': 71, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-4', 'path': 'Twinnings/img', 'startFrame': 19, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-5', 'path': 'Twinnings/img', 'startFrame': 71, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-6', 'path': 'Twinnings/img', 'startFrame': 170, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-7', 'path': 'Twinnings/img', 'startFrame': 231, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-8', 'path': 'Twinnings/img', 'startFrame': 76, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Twinnings-9', 'path': 'Twinnings/img', 'startFrame': 85, 'endFrame': 472, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Twinnings/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-0', 'path': 'Vase/img', 'startFrame': 1, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-1', 'path': 'Vase/img', 'startFrame': 17, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-2', 'path': 'Vase/img', 'startFrame': 33, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-3', 'path': 'Vase/img', 'startFrame': 41, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-4', 'path': 'Vase/img', 'startFrame': 44, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-5', 'path': 'Vase/img', 'startFrame': 122, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-6', 'path': 'Vase/img', 'startFrame': 114, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-7', 'path': 'Vase/img', 'startFrame': 19, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-8', 'path': 'Vase/img', 'startFrame': 130, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Vase-9', 'path': 'Vase/img', 'startFrame': 146, 'endFrame': 271, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Vase/groundtruth_rect.txt', 'object_class': 'other'}
#             ,
#             {'name': 'Walking-0', 'path': 'Walking/img', 'startFrame': 1, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-1', 'path': 'Walking/img', 'startFrame': 33, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-2', 'path': 'Walking/img', 'startFrame': 66, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-3', 'path': 'Walking/img', 'startFrame': 74, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-4', 'path': 'Walking/img', 'startFrame': 132, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-5', 'path': 'Walking/img', 'startFrame': 42, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-6', 'path': 'Walking/img', 'startFrame': 197, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-7', 'path': 'Walking/img', 'startFrame': 58, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-8', 'path': 'Walking/img', 'startFrame': 197, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking-9', 'path': 'Walking/img', 'startFrame': 333, 'endFrame': 412, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-0', 'path': 'Walking2/img', 'startFrame': 1, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-1', 'path': 'Walking2/img', 'startFrame': 16, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-2', 'path': 'Walking2/img', 'startFrame': 81, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-3', 'path': 'Walking2/img', 'startFrame': 16, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-4', 'path': 'Walking2/img', 'startFrame': 41, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-5', 'path': 'Walking2/img', 'startFrame': 26, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-6', 'path': 'Walking2/img', 'startFrame': 61, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-7', 'path': 'Walking2/img', 'startFrame': 106, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-8', 'path': 'Walking2/img', 'startFrame': 201, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Walking2-9', 'path': 'Walking2/img', 'startFrame': 91, 'endFrame': 500, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Walking2/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-0', 'path': 'Woman/img', 'startFrame': 1, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-1', 'path': 'Woman/img', 'startFrame': 12, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-2', 'path': 'Woman/img', 'startFrame': 60, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-3', 'path': 'Woman/img', 'startFrame': 124, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-4', 'path': 'Woman/img', 'startFrame': 119, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-5', 'path': 'Woman/img', 'startFrame': 237, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-6', 'path': 'Woman/img', 'startFrame': 213, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-7', 'path': 'Woman/img', 'startFrame': 42, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-8', 'path': 'Woman/img', 'startFrame': 48, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#             {'name': 'Woman-9', 'path': 'Woman/img', 'startFrame': 213, 'endFrame': 597, 'nz': 4, 'ext': 'jpg',
#              'anno_path': 'Woman/groundtruth_rect.txt', 'object_class': 'person'}
#             ,
#         ]
#
#         return sequence_info_list