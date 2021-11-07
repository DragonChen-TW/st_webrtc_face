from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import streamlit as st
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

import queue

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
# address = 'https://datasci.mis.nsysu.edu.tw:3100/single_jpg'
address = 'http://localhost:3100/single_jpg'

class FlaskFaceDetection:
    def __init__(self):
        self.remote_send = True
        self.session = requests.Session()
        self.session.trust_env = False
        self.sm = SubMeter()
        self.color_set = {}
        self.result_queue = queue.Queue()

    def recv(self, frame):
        self.result_queue = queue.Queue()
        self.result_queue.put(frame)
        ori_img = frame.to_ndarray(format='bgr24') # type is numpy

        # shape is (h, w, 3)
        
        print(ori_img.shape)

        # ----- setting width and height skiped -----

        if self.remote_send:
            print('send')
            self.remote_send = False

            # ----- compress to jpeg -----
            img = sjpg.encode_jpeg(ori_img) #bytes
            # with open('temp.jpg', 'wb') as f:
            #     f.write(img)

            # ----- request backend -----
            res = self.session.post( # -> server_flask
                address,
                files={
                    'picture': ('upload.jpg', io.BytesIO(img)) # Camouflaged file
                },
                headers={'Connection': 'close'},
                verify=False, # For CSIE SSL issue
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
            
            self.remote_send = True

        return av.VideoFrame.from_ndarray(out_img, format='bgr24')

ctx = webrtc_streamer(
    # mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={'video': True, 'audio': False},
    video_processor_factory=FlaskFaceDetection, 
    key='sample',
)

st_show = st.empty()
# st_photo = st.button('Take picture')

# while True:
#     if ctx.video_processor:
#         print('st_photo out:', st_photo)
#         try:
#             video_frame = ctx.video_processor.result_queue.get(timeout=1)
#         except queue.Empty:
#             print('break1')
#             break
#     else:
#         print('break2')
#         break

# if st_photo and video_frame:
#     print('st_photo:', st_photo)
#     video_frame = video_frame.to_ndarray(format='rgb24')
#     st_show.image(video_frame)

if st.button('Take picture'):
    # if ctx.state.playing:
    #     labels_placeholder = st.empty()
    # while True:
    if ctx.video_processor:
        try:
            video_frame = ctx.video_processor.result_queue.get(
                timeout=1.0
            )
        except queue.Empty:
            print('break1')
            # break
        video_frame = video_frame.to_ndarray(format='rgb24')
        st_show.image(video_frame)
    else:
        print('break2')
        # break
