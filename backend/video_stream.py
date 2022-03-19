# !/usr/local/bin/python3
# encodin: utf-8
# author: cx
"""经过测试 cv2.VideoCapture 的 read 函数并不能获取实时流的最新帧
而是按照内部缓冲区中顺序逐帧的读取，opencv会每过一段时间清空一次缓冲区
但是清空的时机并不是我们能够控制的，因此如果对视频帧的处理速度如果跟不上接受速度
那么每过一段时间，在播放时(imshow)时会看到画面突然花屏，甚至程序直接崩溃

在网上查了很多资料，处理方式基本是一个思想
使用一个临时缓存，可以是一个变量保存最新一帧，也可以是一个队列保存一些帧
然后开启一个线程读取最新帧保存到缓存里，用户读取的时候只返回最新的一帧
这里我是使用了一个变量保存最新帧

注意：这个处理方式只是防止处理（解码、计算或播放）速度跟不上输入速度
而导致程序崩溃或者后续视频画面花屏，在读取时还是丢弃一些视频帧

这个在高性能机器上也没啥必要 [/doge]
"""
import gc
import os
import time

import cupy
import numpy as np
from matplotlib import pyplot
from tensorflow.python.keras.backend import set_session

from apps.models.model import detect_save, get_variable_from_model

from apps import STATIC_DIR, app

import threading
import cv2


class Observer:

    def update(self,ok,frame):
        return

    def display(self):
        return


class Subject:

    def registerObserver(self, observer):
        return

    def removeObserver(self, observer):
        return

    def notifyObservers(self):
        return

class RTSCapture(cv2.VideoCapture,Subject,object):
    """Real Time Streaming Capture.
    这个类必须使用 RTSCapture.create 方法创建，请不要直接实例化
    """

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"]  # 用于识别实时流

    @staticmethod
    def create(url, *schemes):
        """实例化&初始化
        rtscap = RTSCapture.create("rtsp://example.com/live/1")
        or
        rtscap = RTSCapture.create("http://example.com/live/1.m3u8", "http://")
        """

        rtscap = RTSCapture(url)
        rtscap.observers = []
        rtscap.frame_receiver = threading.Thread(target=rtscap.notifyObservers, daemon=True)
        rtscap.schemes.extend(schemes)

        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 这里可能是本机设备
            pass

        return rtscap

    def registerObserver(self, observer):
        self.observers.append(observer)
        # print('observers:', self.observers)

        return

    def removeObserver(self, observer):
        self.observers.remove(observer)

        return

    def isStarted(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok



    def notifyObservers(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            # print(type(frame))
            # print(self.observers)
            for item in self.observers:
                # print("update")
                if item.last_access != 0 and time.time() - item.last_access > 30:
                    self.removeObserver(item)
                    del item
                    # gc.collect() #释放循环引用
                    continue
                item.update(ok, frame)
                # item.display()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None

        return frame is not None, frame



    def start_read(self):
        """启动子线程读取视频帧"""
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()


# class Flow_Response (Observer,object):



#     ok = False
#     frame = None
#     last_access = 0

#     @staticmethod
#     def create(weatherData):
#         """实例化&初始化
#         rtscap = RTSCapture.create("rtsp://example.com/live/1")
#         or
#         rtscap = RTSCapture.create("http://example.com/live/1.m3u8", "http://")
#         """

#         rtscap = Flow_Response()
#         weatherData.registerObserver(rtscap)

#         return rtscap

#     def update(self,ok,frame):

#         self.ok = ok
#         self.frame = frame

#         return
#     def display(self):

#         print(self.ok,self.frame)

#         return

#     def get_frame(self):

#         return self.frame

#     def get_bytes(self):
#         # print(self.ok,self.frame)
#         self.last_access = time.time()
#         ret, jpeg = cv2.imencode('.jpg', self.frame)

#         return jpeg.tobytes()

#     def detect(self,detect_relative,response_name):
#         self.last_access = time.time()
#         filename = os.path.join(STATIC_DIR, detect_relative, response_name)
#         graph = get_variable_from_model('graph')
#         sess = get_variable_from_model('sess')
#         model = get_variable_from_model('model')
#         with graph.as_default():
#             set_session(sess)
#             detect_save(self.frame, model, filename)
#         result_frame = pyplot.imread(filename)
#         ret, result_jpeg = cv2.imencode('.jpg', result_frame)

#         return result_jpeg.tobytes()


# video_subject = RTSCapture.create(app.config['video'],"http://")
# video_subject.start_read()

# # while True:
# #     url_observer.video()
# def feed_gen():
#     global video_subject
#     url_observer = Flow_Response.create(video_subject)
#     while True:

#         # 帧处理代码写这里
#         frame = url_observer.get_frame()
#         # url_observer.display()
#         try:
#             if frame == None:
#                 continue
#         except:
#             bytes = url_observer.get_bytes()

#             # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + bytes + b'\r\n\r\n')



# def out_gen(detect_relative,response_name):
#     global video_subject
#     url_observer = Flow_Response.create(video_subject)
#     while True:

#         # 帧处理代码写这里
#         frame = url_observer.get_frame()
#         # url_observer.display()
#         try:
#             if frame == None:
#                 continue
#         except:
#             bytes = url_observer.detect(detect_relative,response_name)

#             # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + bytes + b'\r\n\r\n')


