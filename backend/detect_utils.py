import numpy as np
def pngMaskJpg(jpg, png, color="green"):
    color_list = {
        'blue': [255, 0, 0],
        'red': [0, 0, 255],
        'green': [0, 255, 0],
        'yellow':[44,208,245],
        'pink':[217,100,252],
    }

    empty_image = np.zeros_like(jpg)
    if isinstance(color, str):
        if color not in color_list.keys():
            color = 'green'
        color = color_list[color]
    empty_image[png == 1] = color

    print(jpg.shape, empty_image.shape)
    jpg[png!=0] = empty_image[png!=0] * 0.5 + jpg[png!=0]*0.5

    return jpg
