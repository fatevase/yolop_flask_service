"""经过测试 cv2.VideoCapture 的 read 函数并不能获取实时流的最新帧
而是按照内部缓冲区中顺序逐帧的读取，opencv会每过一段时间清空一次缓冲区
但是清空的时机并不是我们能够控制的，因此如果对视频帧的处理速度如果跟不上接受速度
那么每过一段时间，在播放时(imshow)时会看到画面突然花屏，甚至程序直接崩溃
注意：这个处理方式只是防止处理（解码、计算或播放）速度跟不上输入速度
而导致程序崩溃或者后续视频画面花屏，在读取时还是丢弃一些视频帧
"""
import time
import threading
import cv2


class Observer:
    def update(self, ok, frame):
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
            # 网络流
            rtscap._reading = True
        elif isinstance(url, int):
            # 本机摄像头
            rtscap._reading = True

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


class StreamObserver(Observer, object):
    ok = False
    frame = None
    last_access = 0

    @staticmethod
    def create(stream_subject):
        stream_observer = StreamObserver()
        stream_subject.registerObserver(stream_observer)
        return stream_observer
    
    def update(self, ok, frame):
        self.ok = ok
        self.frame = frame

    def display(self):
        print(self.ok, self.frame)
        if self.ok:
            cv2.imshow("StreamObserver", self.frame)
            cv2.waitKey(1)
        
    def get_frame(self):
        self.last_access = time.time()
        return self.frame
    
    def get_bytes(self):
        self.last_access = time.time()
        _, jpg = cv2.imencode('.jpg', self.frame)
        return jpg.tobytes()
    
    def detect():
        pass



if __name__ == "__main__":
    # video_subject = RTSCapture.create('http://vfx.mtime.cn/Video/2019/02/04/mp4/190204084208765161.mp4', 'http://')
    video_subject = RTSCapture.create(0)
    video_subject.start_read()
    url_observer = StreamObserver.create(video_subject)


    while True:

        frame = url_observer.get_frame()
        if frame is not None:
            url_observer.display()
        
        # try:
        #     if frame == None:
        #         continue
        # except:
        #     url_observer.display()
            # video_subject.stop_read()
