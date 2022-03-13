from crypt import methods
from flask import Flask, jsonify, render_template, request, send_from_directory, make_response
import os, io
import time
import base64
import cv2


from modules.detect import detect_from_img

class WebUtils:
    def __init__(self):
        pass
    @staticmethod
    def fileUpload(file, file_dir):
        ALLOWED_EXTENSIONS = set([
            'txt', 'png', 'jpg', 'xls', 'JPG', 'PNG',
            'xlsx', 'gif', 'GIF'])  # 允许上传的文件后缀
        def allowedFile(filename):
            return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)  # 文件夹不存在就创建
        f = file # 传入的文件
        if f and allowedFile(f.filename):  # 判断是否是允许上传的文件类型
            fname = f.filename
            ext = fname.rsplit('.', 1)[1]  # 获取文件后缀
            unix_time = int(time.time())
            new_filename = str(unix_time) + '.' + ext  # 修改文件名
            f.save(os.path.join(file_dir, new_filename))  # 保存文件到upload目录
            return {
                    "file_dir": os.path.join(file_dir, new_filename)
            }
        else:
            return {
                    "file_dir": -1
            }

    # 压缩base64
    @staticmethod
    def compress_image(img, kb=190, k=0.9):
        """不改变图片尺寸压缩到指定大小
        :param outfile: 压缩文件保存地址
        :param kb: 压缩目标，KB
        :param step: 每次调整的压缩比率
        :param quality: 初始压缩比率
        :return: 压缩文件地址，压缩文件大小
        """
        o_size = ((len(base64.b64encode(img)) +2 )*4 / 3)//1024
        out = img
        while o_size > kb:
            x, y, _ = img.shape
            out = cv2.resize(out, (int(x * k), int(y * k)), interpolation = cv2.INTER_AREA) 
            o_size =((len(base64.b64encode(out)) +2 )*4 / 3)//1024
            img = out
        print(img.shape)
        return img

    @staticmethod
    def tryRoadDetect(img, img_type='base64',
            need_lane=False, need_road=False, need_car=False):
        data = detect_from_img(img, img_type=img_type,
        weight_file='modules/YOLOP/weights/End-to-end.pth')
        result = dict()
        # lane = WebUtils.compress_image(data['lane'])
        # ego = WebUtils.compress_image(data['ego'])
        # car_bbox = WebUtils.compress_image(data['car_bbox'])
        if need_lane:
            lane = data['lane']
            result['lane'] = base64.b64encode(lane).decode('utf-8')
        if need_road:
            ego = data['ego']
            result['road'] = base64.b64encode(ego).decode('utf-8')
        if need_car:
            car_bbox = data['car_bbox']
            bbox = data['bbox_det']
            result['car_bbox'] = base64.b64encode(car_bbox).decode('utf-8')
            result['bbox'] = bbox

        return result


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


    def jsonifyResult(self, code=0, data=[]):
        return jsonify({"code":code, "data":data})  

    def home(self):
        return render_template('index.html')
    
    def form(self):
        return self.jsonifyResult(data=request.values)

    def roadDetector(self, code=0, data=[]):
        request_file = request.files
        raw_img = request_file['img']
        img_b64 = base64.b64encode(raw_img.read())
        data = WebUtils.tryRoadDetect(img_b64, need_car=True)
        return jsonify({"code":code, "data":data})

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
