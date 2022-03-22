from crypt import methods
from flask import Flask, jsonify, render_template, request, send_from_directory, make_response, Response
import os, io
import json
import time
import base64
import cv2, numpy
import re
import random

from modules.detect import detect_from_img
from test_socket import testSocket
from web_utils import WebUtils
from video_stream import RTSCapture, StreamObserver

class AppClass():
    '''
    flask app wrapped in class 
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        self.UPLOAD_FOLDER = './uploads'
        self.basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前项目的绝对路径
        super().__init__()
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = self.UPLOAD_FOLDER  # 设置文件上传的目标文件夹
        app = self.app
        
        #
        # setup global handlers
        #
        @app.before_first_request
        def before_first_request_func():
            pass
        
        #
        # setup the RESTFUL routes for this application
        #
        @app.route('/')
        def index():
            return self.home()
        
        @app.route('/form', methods=['GET', 'POST'])
        def test_form():
            print(request.args)
            return self.form()
        
        @app.route('/api/upload', methods=['GET', 'POST'])
        def test_upload():
            return self.upload()

        @app.route('/api/download', methods=['GET', 'POST'])
        # file download
        def try_download():
            return self.downloader(request)

        @app.route('/api/detect', methods=['GET', 'POST'])
        def road_detect():
            return self.roadDetector()

        @app.route('/api/car_info', methods=['GET', 'POST'])
        def car_info():
            return self.getCarInfo()
        
        @app.route('/detect_stream', methods=['GET','POST'])
        def stream_detect():
        #only for json data
            stream = self.streamDetect(self.basedir+'/inference/videos/1.mp4')
            return self.streamResult(stream)

        @app.route('/video_test', methods=['GET', 'POST'])
        def video_test():
            stream = WebUtils.test_gen(self.basedir+'/inference/videos/1.mp4')
            return self.streamResult(stream)

    def jsonifyResult(self, code=0, data=[]):
        return jsonify({"code":code, "data":data})
    
    def streamResult(self, stream=[], mimetype='multipart/x-mixed-replace; boundary=frame'):
        return Response(stream, mimetype=mimetype)

    def home(self):
        return render_template('index.html')
    
    def form(self):
        return self.jsonifyResult(data=request.values)
    
    def streamDetect(self, source):

        if not hasattr(self, 'video_subject'):
            self.video_subject = RTSCapture.create(source)
            self.video_subject.start_read()
        url_observer = StreamObserver.create(self.video_subject)
        print(url_observer, self.video_subject)
        # if not hasattr(self, 'detect_args'):
        #     self.detect_args = dict()
        #     self.detect_args['img_type'] = 'stream'

        vid = cv2.VideoCapture(source)
        while True:
            return_value, frame = vid.read()
            detect_args = self.detect_args

            data = WebUtils.tryRoadDetect(frame, **detect_args)
            output = data['output']
            image = output.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

        # while True:
        #     frame = url_observer.get_frame()
        #     # print(frame)
        #     if frame is not None:
        #         detect_args = self.detect_args
        #         print(frame)
        #         data = WebUtils.tryRoadDetect(frame, **detect_args)
        #         output = data['output']
        #         image = cv2.imencode('.jpg', output)[1].tobytes()
        #         yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    def roadDetector(self, code=1, data=[]):

        detect_args = dict()
        try:
            detect_list = request.values.get('detect_list')

            if detect_list is not None:
                detect_list = json.loads(detect_list)
            
                if detect_list['lane_detection']:
                    detect_args['need_lane'] = True
                if detect_list['ego_detection']:
                    detect_args['need_road'] = True
                if detect_list['congestion_detection']:
                    detect_args['need_car'] = True

            data_type = request.values.get('data_type')
            if data_type == 'file':
                request_file = request.files
                raw_img = request_file['source']
                source = base64.b64encode(raw_img.read())
            elif data_type == 'base64':
                source = request.values.get('source')
            elif data_type == 'stream':
                source = request.values.get('source')
                self.detect_args = detect_args
                self.detect_args['img_type'] = 'stream'

                data = dict(output="http://localhost:5001/detect_stream")
                print(data)
                return jsonify({'code':code, "data": data})
            # waster most was detect in model,
            # time 1.2s=>0.2s
            data = WebUtils.tryRoadDetect(source, **detect_args)
            return jsonify({"code":code, "data":data})
        except Exception as e:
            print(e)
            return jsonify({"code":-1, "data":str(e)})
        
    def getCarInfo(self):
        data = dict()
        data['gas_num'] = 80 + (random.randint(0,30)-10)
        data['car_speed'] = 60 + (random.randint(0,40)-20)
        data['sensor_num'] = WebUtils.get_cam_num()
        data['main_camera'] = f"camera-{0}"
        data['car_temperature'] = 30 + (random.randint(0,100)-50) / 10.0
        data['car_humidity'] = 50 + (random.randint(0,20)-10)
        data['car_pollute'] = random.randint(0, 10) / 100.0
        socket_data = testSocket('127.0.0.1:5001')
        data['net_delay'] = -1
        if socket_data['code'] > 0:
            data['net_delay'] = socket_data['dems']
        return jsonify({"code":socket_data['code'], "data":data})


    def downloader(self, data):
        dirpath = os.path.join(self.app.root_path, self.UPLOAD_FOLDER)  # 这里是下在目录，从工程的根目录写起，比如你要下载static/js里面的js文件，这里就要写“static/js”
        # return send_from_directory(dirpath, filename, as_attachment=False)  # as_attachment=True 一定要写，不然会变成打开，而不是下载
        file_name = data.args['filename']
        img_file = open(os.path.join(dirpath, file_name), "rb").read()
        img_base64 = base64.b64encode(img_file)
        return self.responseData(base64.b64decode(img_base64))
        #return send_from_directory(dirpath, file_name, as_attachment=False)  # as_attachment=True  下载
    
    def upload(self):
        data = request
        print(request.data)
        print("---"*10)
        file = request.files['myfile']
        save_dir = os.path.join(self.basedir, self.app.config['UPLOAD_FOLDER'])
        result = WebUtils.fileUpload(file, save_dir)
        return self.jsonifyResult(data=result)

    def run(self, host="0.0.0.0", port=5001, debug=True):
        self.app.run(host=host, port=port, debug=debug)
        return self.app

    def responseData(self, data,  content_type='image/jpeg'):
        resp = make_response(data)
        #设置response的headers对象
        resp.headers['Content-Type'] = content_type
        # resp.headers['image-focus'] = 14521.6
        return resp


# if __name__ == '__main__':
app_class = AppClass()
core_app = app_class.run()
