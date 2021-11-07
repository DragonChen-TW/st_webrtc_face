from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
# 
from PIL import Image
import cv2
import simplejpeg as sjpg
import requests
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()
import random
import os
import io
import time
import logging
import numpy as np
# 
from meter import SubMeter
from plotting import draw, names

# import http.client
# http.client.HTTPConnection.debuglevel = 1

def print_to_file(*args):
    out_arr = [str(a) for a in args]
    with open('out.txt', 'a+') as f:
        f.write(' '.join(out_arr))
        f.write('\n')
# print = print_to_file

# variable setting
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# address = 'https://140.117.75.46:2087/detect_img_jpg'
# address = 'https://192.168.1.7:2087/detect_img_jpg'

# simple YOLOv5
# address = 'http://192.168.1.6:3100/single_jpg'
# address = 'http://140.117.75.46:3100/single_jpg'
# address = 'https://140.117.75.46:3100/single_jpg'
address = 'https://datasci.mis.nsysu.edu.tw:3100/single_jpg'
# address = 'https://localhost:3100/single_jpg'

class FlaskFaceDetection:
    def __init__(self):
        self.remote_send = True
        self.session = requests.Session()
        self.session.trust_env = False
        self.sm = SubMeter()
        self.response = SubMeter()

        self.color_set = {}

    def recv(self, frame):
        ori_img = frame.to_ndarray(format='bgr24') # type is numpy
        # shape is (h, w, 3)
        
        print(ori_img.shape)

        # ----- setting width and height skiped -----

        if self.remote_send:
#             print('send')
            self.remote_send = False
            
            t1 = time.time()

            # ----- compress to jpeg -----
            img = sjpg.encode_jpeg(ori_img)
            # with open('temp.jpg', 'wb') as f:
            #     f.write(img)

            # ----- request backend -----
            res = self.session.post(
                address,
                files={
                    'picture': ('upload.jpg', io.BytesIO(img))
                },
                headers={'Connection': 'close'},
                verify=False,
            )
            
            # # ----- draw -----
            # res = res.json()
            # print(res)
            # self.sm.times_update(res['time'])
            # self.sm.plot()

            # is_identified = [r['name'] for r in res['registered']]

            # out_img = ori_img

            # for n in is_identified:
            #     if n not in self.color_set:
            #         self.color_set[n] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # for i, face in enumerate(res['faces']):
            #     p1 = tuple(face[0])
            #     p2 = tuple(face[1])
            #     c = self.color_set[is_identified[i]]
            #     img = cv2.rectangle(out_img, p1, p2, c, 10)

            # for reg in res['registered']:
            #     text = '{:.02f} {}'.format(reg['confidence'], reg['name'])
            #     f_idx = reg['face_id']
            #     loc = res['faces'][f_idx][0]
            #     loc = (loc[0], loc[1] - 10)
            #     c = self.color_set[reg['name']]
            #     cv2.putText(out_img, text, loc, cv2.FONT_HERSHEY_DUPLEX, 1, c, 2, cv2.LINE_AA)


            # draw for .6 YOLOv5
            res = res.json()
#             print('res', res)

            class Res:
                names = names
            res_cls = Res()
            for k, v in res.items():
                setattr(res_cls, k, v)
            
            self.sm.update(res['encode_time'], res['modeling_time'], 1e-10, 1e-10)
            self.sm.plot()
            if len(self.sm.total_list) == 30:
                print('fps', 1 / (sum(self.sm.total_list) / len(self.sm.total_list)))
                logging.warning(f'avg {np.mean(self.sm.total_list)}')
                logging.warning(f'fps {1 / np.mean(self.sm.total_list)}')
            
            t1 = time.time() - t1
            self.response.update(t1, 1e-10, 1e-10, 1e-10)
            self.response.plot()
            if len(self.response.total_list) == 30:
                print('fps', 1 / (sum(self.response.total_list) / len(self.response.total_list)))
                logging.warning(f'avg {np.mean(self.response.total_list)}')
                logging.warning(f'fps {1 / np.mean(self.response.total_list)}')


            out_img = ori_img
            out_img = draw(res_cls, [ori_img])[0]
            
            self.remote_send = True

        return av.VideoFrame.from_ndarray(out_img, format='bgr24')

webrtc_streamer(
    # mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={'video': True, 'audio': False},
    video_processor_factory=FlaskFaceDetection,
    key='sample',
)

