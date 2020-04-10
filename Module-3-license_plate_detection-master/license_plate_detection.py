import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import os

files = [
    {'name': 'car_1.jpg', 'color': 'white'},
    {'name': 'car_2.jpg', 'color': 'special', 'option':{
        'type': 'color',
        'lower_white': np.array([0,0,200], dtype=np.uint8),
        'upper_white': np.array([255,30,255], dtype=np.uint8),
        'no_bitwise': False
    }},
    {'name': 'car_3.jpg', 'color': 'yellow'},
    {'name': 'car_4.jpg', 'color': 'white_bg'},
    {'name': 'car_5.jpg', 'color': 'white'},
    {'name': 'car_6.jpg', 'color': 'white'},
    {'name': 'car_7.jpg', 'color': 'white'},
    {'name': 'car_8.jpg', 'color': 'yellow'},
    {'name': 'car_9.jpg', 'color': 'white'},
    {'name': 'car_10.jpg', 'color': 'white'},
    {'name': 'car_11.jpg', 'color': 'white'},
    {'name': 'car_12.jpg', 'color': 'yellow'},
    {'name': 'car_13.jpg', 'color': 'yellow'},
    {'name': 'car_14.jpg', 'color': 'yellow'}
]

def img_process(ori_img, color, option):
    image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(image, thresh=200, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if option is not None and option['no_bitwise']:
        img_mask = image
    else:
        img_mask = cv2.bitwise_and(image, mask)

    if color in ('white', 'yellow'):
        hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
        h, s, v1 = cv2.split(hsv)
        if color == 'white':
            lower_white = np.array([0,0,150], dtype=np.uint8)
            upper_white = np.array([255,40,255], dtype=np.uint8)
        elif color == 'yellow':
            lower_white = np.array([20, 100, 100], dtype=np.uint8)
            upper_white = np.array([30, 255, 255], dtype=np.uint8)
        res_mask = cv2.inRange(hsv, lower_white, upper_white)
        res_img = cv2.bitwise_and(v1, image, mask=res_mask)
    else:
        if color == 'special' and option is not None and option['type'] == 'color':
            print(":::::Special - color")
            hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
            h, s, v1 = cv2.split(hsv)
            upper_white = option['upper_white']
            lower_white = option['lower_white']
            res_mask = cv2.inRange(hsv, lower_white, upper_white)
            res_img = cv2.bitwise_and(v1, image, mask=res_mask)
        else:
            res_img = img_mask

    return res_img

def contour(res_img, result, color):
    # Contours
    contours, _ = cv2.findContours(
            res_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    NumberPlateCnt = None
    found = False
    lt, rb = [10000, 10000], [0, 0]

    if color == 'white_bg':
        for c in contours:
             peri = cv2.arcLength(c, True)
             approx = cv2.approxPolyDP(c, 0.06 * peri, True)
             if len(approx) == 4:
                 found = True
                 NumberPlateCnt = approx
                 break
        if found:
            cv2.drawContours(result, [NumberPlateCnt], -1, (255, 0, 255), 2)

            for point in NumberPlateCnt:
                cur_cx, cur_cy = point[0][0], point[0][1]
                if cur_cx < lt[0]: lt[0] = cur_cx
                if cur_cx > rb[0]: rb[0] = cur_cx
                if cur_cy < lt[1]: lt[1] = cur_cy
                if cur_cy > rb[1]: rb[1] = cur_cy

            cv2.circle(result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
            cv2.circle(result, (rb[0], rb[1]), 2, (150, 200, 255), 2)

            crop = res_img[lt[1]:rb[1], lt[0]:rb[0]]
        else:
            crop = res_img.copy()
    elif len(contours) > 0:
        # Convex Hull
        hull = cv2.convexHull(contours[0])
        approx2 = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
        cv2.drawContours(result, [approx2], -1, (255, 0, 255), 2, lineType=8)

        for point in approx2:
            cur_cx, cur_cy = point[0][0], point[0][1]
            if cur_cx < lt[0]: lt[0] = cur_cx
            if cur_cx > rb[0]: rb[0] = cur_cx
            if cur_cy < lt[1]: lt[1] = cur_cy
            if cur_cy > rb[1]: rb[1] = cur_cy

        cv2.circle(result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
        cv2.circle(result, (rb[0], rb[1]), 2, (150, 200, 255), 2)

        crop = res_img[lt[1]:rb[1], lt[0]:rb[0]]
    else:
        crop = res_img.copy()
    return crop

def show_plt(crop_img, img_name, directory, flag):

    plt.subplot(235),plt.imshow(crop_img, cmap = 'gray')
    plt.title('Number Plate Cropped'), plt.xticks([]), plt.yticks([])
    plt.suptitle(img_name)

    if flag:
        plt.show()

    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite("{}/{}".format(directory, img_name), crop_img)

    return pytesseract.image_to_string(crop_img, config='--psm 11')

def main(flag):
    source_directory = "input"
    license_number = {}
    for file in files:
        print("License Plates - {}".format(file['name']))
        # 1.	Read the car images with visible number plates
        ori_img = cv2.imread("{}/{}".format(source_directory, file['name']))
        option = None
        if file['color'] == 'special':
            option = file['option']
        img_name = file['name']
        result = ori_img

        # 2.	Process the image by filtering using different threshold for white and yellow plates
        res_img = img_process(ori_img.copy(), file['color'], option)

        '''
            3.	Mark contours around the number plate
            4.	Crop the portion of the image with number plate
        '''
        crop_img = contour(res_img, result, file['color'])

        '''
            6.	Use pytesseract to extract the text from the cropped image
            7.	Write the cropped image to output folder
        '''
        number_text = show_plt(crop_img, img_name, "output", flag)

        # 8.	Store the image to license plate number  mapping in dictionary
        license_number[img_name] = number_text

    print(license_number)

if __name__ == '__main__':
    flag = True
    main(flag)
