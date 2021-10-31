import matplotlib.pyplot as plt

class SubMeter:
    def __init__(self):
        self.savefile_list = []
        self.yolo_list = []
        self.mobile_list = []
        self.plda_list = []
        self.total_list = []
    
    def update(self, s1, s2, s3):
        self.yolo_list.append(s1)
        self.mobile_list.append(s2)
        self.plda_list.append(s3)
        self.total_list.append(s1 + s2 + s3)
    
    def times_update(self, times):
        self.savefile_list.append(times['save_file_time'])
        self.yolo_list.append(times['yolo_time'])
        self.mobile_list.append(times['mobilenet_time'])
        self.plda_list.append(times['plda_time'])
        self.total_list.append(sum([times[k] for k in times]))
    
    def plot(self, outfile=None):
        if len(self.total_list) == 30:
            print('-' * 10, 'modeling summary', '-' * 10)
            print('round', len(self.total_list), 'avg savefile', sum(self.savefile_list) / len(self.savefile_list))
            print('round', len(self.total_list), 'avg yolov5', sum(self.yolo_list) / len(self.yolo_list))
            print('round', len(self.total_list), 'avg mobilenet', sum(self.mobile_list) / len(self.mobile_list))
            print('round', len(self.total_list), 'avg PLDA', sum(self.plda_list) / len(self.plda_list))
            print('round', len(self.total_list), 'avg', sum(self.total_list) / len(self.total_list))