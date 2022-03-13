from modules.detect import detect_from_img
import base64
if __name__ == "__main__":
    file_name = 'inference/images/0ace96c3-48481887.jpg'
    img_file = open(file_name, "rb").read()
    img_base64 = base64.b64encode(img_file)
    result = detect_from_img(img_base64, img_type='base64',
    weight_file='modules/YOLOP/weights/End-to-end.pth')