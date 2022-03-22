import cv2
import numpy as np


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy+ ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), 1, 4)

def passby_detect(origin_img):
    img = origin_img
    passby_mask = np.zeros(img.shape[:2], np.uint8)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(img, 0, (8, 8), (32, 32), 1.05, 2)

    found_filtered = []
    passby_bbox = []
    for ri, r in enumerate(found):
        for qi, q, in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
            else:
                found_filtered.append(r)
        for person in found_filtered:
            draw_person(passby_mask, person)
            passby_bbox.append(person.tolist())
    return passby_mask, passby_bbox