from crypt import methods
import os, io
import json
import base64
import cv2
import numpy as np

from modules.detect import detect_from_img
import  detect_utils

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
    def get_cam_num(max_cam_num=10):
        cnt = 0
        for device in range(0, max_cam_num):
            stream = cv2.VideoCapture(device)

            grabbed = stream.grab()
            stream.release()
            if not grabbed:
                break
            cnt = cnt + 1
        return cnt
    
    @staticmethod
    def tryRoadDetect(img, img_type='base64',
            need_lane=False, need_road=False, need_car=False):
        if img_type == 'base64':
            img = img.split(';base64,')[-1]

        data = detect_from_img(img, img_type=img_type,
        weight_file='modules/YOLOP/weights/End-to-end.pth')
        result = dict()

        output = data['raw_img']
        # dst = np.zeros(output.shape, dtype=np.float32)
        # output = cv2.normalize(output, None, alpha=0, beta=1,
        #                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # print(np.unique(output, return_counts=True))
        if need_lane:
            lane = data['lane']
            output = detect_utils.pngMaskJpg(output, lane, color="blue")

            # output = output + lane

        if need_road:
            ego = data['ego']
            output = detect_utils.pngMaskJpg(output, ego, color="green")

        if need_car:
            car_bbox = data['car_bbox']
            result['bbox_list'] = data['bbox_list']
            print(f"car_bbox shape: {car_bbox.shape}, output: {output.shape}")
            output = detect_utils.pngMaskJpg(output, car_bbox, color="yellow")
            outpuy = car_bbox

        print(f"--------{output.shape}")
        retval, output = cv2.imencode('.png', output)
        result['output'] = base64.b64encode(output).decode('utf-8')
        return result

