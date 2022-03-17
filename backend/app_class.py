from crypt import methods
from flask import Flask, jsonify, render_template, request, send_from_directory, make_response
import os, io
import json
import time
import base64
import cv2, numpy
import re

from modules.detect import detect_from_img
from test_socket import testSocket
from web_utils import WebUtils

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
            # img 传入
            # print('---'*10)
            # print(request.files)
            return self.roadDetector()

        @app.route('/api/car_info', methods=['GET', 'POST'])
        def car_info():
            return self.getCarInfo()

    def jsonifyResult(self, code=0, data=[]):
        return jsonify({"code":code, "data":data})  

    def home(self):
        return render_template('index.html')
    
    def form(self):
        return self.jsonifyResult(data=request.values)

    def roadDetector(self, code=0, data=[]):
        try:
            data_type = request.values.get('data_type')
            if data_type == 'file':
                request_file = request.files
                raw_img = request_file['img']
                img_b64 = base64.b64encode(raw_img.read())
            elif data_type == 'base64':
                img_b64 = request.values.get('img')

            detect_list = request.values.get('detect_list')
            detect_list = json.loads(detect_list)

            detect_args = dict()
            if detect_list['lane_detection']:
                detect_args['need_lane'] = True
            if detect_list['ego_detection']:
                detect_args['need_road'] = True
            if detect_list['congestion_detection']:
                detect_args['need_car'] = True
            
            print(detect_args)
            data = WebUtils.tryRoadDetect(img_b64, **detect_args)
            return jsonify({"code":code, "data":data})
        except Exception as e:
            return jsonify({"code":-1, "data":e})
        
    def getCarInfo(self):
        data = dict()
        data['speed'] = 120
        data['device'] = WebUtils.get_cam_num()
        socket_data = testSocket('127.0.0.1:5001')
        data['dems'] = -1
        if socket_data['code'] > 0:
            data['dems'] = socket_data['dems']
        return jsonify({"code":1, "data":data})


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
