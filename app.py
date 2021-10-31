from streamlit_webrtc import webrtc_streamer
import av
# 
from PIL import Image
import cv2
import simplejpeg as sjpg
import requests
import random
import os
import io
import time
# 
from meter import SubMeter
from plotting import draw, names

import http.client
http.client.HTTPConnection.debuglevel = 1

def print_to_file(*args):
    out_arr = [str(a) for a in args]
    with open('out.txt', 'a+') as f:
        f.write(' '.join(out_arr))
        f.write('\n')
# print = print_to_file

# variable setting
# address = 'https://140.117.75.46:2087/detect_img_jpg'
# address = 'https://192.168.1.7:2087/detect_img_jpg'

# simple YOLOv5
# address = 'http://192.168.1.6:3100/single_jpg'
# address = 'http://140.117.75.46:3100/single_jpg'
address = 'https://140.117.75.46:3100/single_jpg'

class GPRCFaceDetection:
    def __init__(self):
        self.grpc_send = True
        self.session = requests.Session()
        self.session.trust_env = False
        self.sm = SubMeter()

        self.color_set = {}

    def recv(self, frame):
        ori_img = frame.to_ndarray(format='bgr24') # type is numpy
        # shape is (h, w, 3)
        
        print(ori_img.shape)

        # ----- setting width and height skiped -----

        if self.grpc_send:
            print('send')
            self.grpc_send = False

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
            print('res', res)

            class Res:
                pass
            res_cls = Res()
            res_cls.names = names
            for k, v in res.items():
                setattr(res_cls, k, v)

            out_img = ori_img
            out_img = draw(res_cls, [ori_img])[0]
            
            self.grpc_send = True

        return av.VideoFrame.from_ndarray(out_img, format='bgr24')

webrtc_streamer(
    media_stream_constraints={'video': True, 'audio': False},
    video_processor_factory=GPRCFaceDetection,
    key='sample',
)

