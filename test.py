# import necessary packages
import numpy as np
import cv2
import os
from pytesseract import image_to_string
import shutil
import imutils
import time
import random
import math
from classifier import keyword_detection, sentence_classifier
from webcolors import name_to_rgb
import pickle

dict_label_color = {'SHIPPER': "blue", 'CONSIGNEE': "green", 'NOTIFY': "red", 'ALSO_NOTIFY': "magenta", 'POR': "yellow", 'POL': "cyan",
                    'POD': "navy", 'DEL': "pink", 'DESCRIPTION': "purple", 'VESSEL': "gray", 'Gross Weight': "lavender", 'Measurement': "orange"}


# take only character boxes

def take_character_boxes(image):
    # find contours
    _, contours, _ = cv2.findContours(
        image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape
    output = np.zeros((height, width), np.uint8)

    # loop in all the contour areas
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # if (w > 5 * h and h < 10) or (h > 5 * w and w < 10):
        #     continue
        if (w > 5 and w < 200) and (h > 5 and h < 70):
            output[y:y+h, x:x+w] = image[y:y+h, x:x+w]

    return output, contours

# processing line advanced by line boxing


def process_line_advanced(thresh, output, binary_img):

    thresh, _ = take_character_boxes(thresh)
    cv2.imwrite("test/s3_boxes.jpg", thresh)

    kernel2 = np.ones((1, 60), np.uint8)
    line_img = cv2.dilate(thresh, kernel2, iterations=1)
    cv2.imwrite("test/s4_textlines.jpg", line_img)

    _, contours, _ = cv2.findContours(
        line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_kv = output.copy()
    output2 = output.copy()
    height, width, _ = output.shape
    # make backround white
    white_bg = np.ones((height, width, 3), np.uint8) * 255
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_black = (0, 0, 0)

    # dict keyword
    # dict_key_unique = {}
    # list of boxes info
    list_boxes_info = []

    # statistic keyword
    list_ocrstring = []

    for index, cnt in enumerate(contours):
        # try crop text region
        x, y, w, h = crop_contours(thresh, cnt)
        if w < 10 or h < 10 or h > 100:
            continue

        # ocr text box
        crop_img = output2[y - 3:y + h + 3, x - 3:x + w + 3]
        ocr_result = image_to_string(
            crop_img, config='-l eng --tessdata-dir "tessdata" --psm 13').lower()

        list_ocrstring.append(ocr_result)

        # flag key or not
        flag_key = False
        textbox_content = ocr_result

        # detect key or not
        try:
            keyword = keyword_detection(ocr_result)
            label = keyword[0]
            percent = keyword[1]
            if label != None:
                # if label not in dict_key_unique:
                #     dict_key_unique[label] = (percent, x, y, w, h)
                # else:
                #     if percent < dict_key_unique[label][0]:
                #         dict_key_unique[label] = (percent, x, y, w, h)
                cv2.putText(output, str(label) + " : " + str(percent), (x, y-3),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color_blue, thickness=2)
                cv2.putText(white_bg, str(label), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=color_black, thickness=2)
                flag_key = True
                textbox_content = label

        except:
            print("can not detect key")
        print(str(index) + " : " + ocr_result)
        # append box to list boxes
        box_color = (0, 0, 0)
        if flag_key == True:
            box_color = name_to_rgb(dict_label_color[textbox_content])

        list_boxes_info.append(
            (x, y, x+w, y+h, flag_key, textbox_content, box_color))

        cv2.rectangle(white_bg, (x, y), (x + w, y + h),
                      color=color_black, thickness=2)
        cv2.rectangle(output, (x - 3, y - 3),
                      (x + w + 3, y + h + 3), color=color_red, thickness=2)
        cv2.putText(output, str(index), (x+w, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color_red, thickness=2)

        # if index == 132:
        #     cv2.imwrite("test/132.png", crop_img)

    # make key value image
    image_key_value = make_key_value(binary_img, list_boxes_info, output2)

    opacity = 0.5
    cv2.addWeighted(image_key_value, opacity, img_kv,
                    1 - opacity, 0, img_kv)

    return output, white_bg, img_kv, list_ocrstring


def make_key_value(binary_img, list_boxes_info, image_key_value):
    # find all keys
    key_list = []
    for index, item in enumerate(list_boxes_info):
        (xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color) = item
        if flag_key == True:
            key_list.append(item)

    if len(key_list) == 0:
        return image_key_value

    for item in key_list:
        find_value_by_key(binary_img, item, list_boxes_info)

    # test save image
    for index, item in enumerate(list_boxes_info):
        (xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color) = item
        if box_color == (0, 0, 0):
            continue
        cv2.rectangle(image_key_value, (xmin, ymin),
                      (xmax, ymax), color=box_color, thickness=cv2.FILLED)
        if flag_key == True:
            cv2.putText(image_key_value, str(textbox_content), (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=box_color, thickness=2)

    return image_key_value


def find_value_by_key(binary_img, key, list_boxes_info):

    # find nearest below box, check it is key or value
    sorted_list_boxes_candidate = find_listboxes_below(
        list_boxes_info, key, thres_range=50)

    if len(sorted_list_boxes_candidate) == 0:
        return
    else:
        (xmin, ymin, xmax, ymax, flag_key, textbox_content,
         box_color, index) = sorted_list_boxes_candidate[0]
        if flag_key == False:
            # dectect vertical value
            detect_vertical_values(list_boxes_info, key,
                                   binary_img, thres_range=200)
        else:
            new_binary_img = binary_img.copy()
            h, w = new_binary_img.shape
            cv2.rectangle(new_binary_img, (0, ymin - 10), (w, ymin - 9),
                          color=(255, 255, 255), thickness=cv2.FILLED)
            (xmin_k, ymin_k, xmax_k, ymax_k, flag_key_k,
             textbox_content_k, box_color_k) = key

            key_inverse = find_key_inverse(
                list_boxes_info, key, thres_range=10)
            detect_vertical_values(
                list_boxes_info, key_inverse, new_binary_img, thres_range=50)


def find_key_inverse(list_boxes_info, key, thres_range=10):
    (xmin_k, ymin_k, xmax_k, ymax_k, flag_key_k,
     textbox_content_k, box_color_k) = key

    list_boxes_candidate = []
    for index, item in enumerate(list_boxes_info):
        (xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color) = item
        if ((np.absolute(ymin - ymin_k) < thres_range) and (xmax_k < xmin)):
            list_boxes_candidate.append(
                (xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color))

    sorted_list_boxes_candidate = sorted(
        list_boxes_candidate, key=lambda tup: (tup[0], tup[1]))

    if(len(sorted_list_boxes_candidate) > 0):
        (xmin, ymin, xmax, ymax, flag_key, textbox_content,
         box_color) = sorted_list_boxes_candidate[0]
        key_inverse = (xmin, 2*ymin - ymax - 5, xmax, ymin - 5,
                       flag_key_k, textbox_content_k, box_color_k)
        return key_inverse

    return key


def find_listboxes_below(list_boxes_info, key, thres_range):
    (xmin_k, ymin_k, xmax_k, ymax_k, flag_key_k,
     textbox_content_k, box_color_k) = key

    # if textbox_content_k == "POD":
    #     print("debug")

    # calculate box range of key
    v_thres_min, v_thres_max = calculate_key_range(
        binary_img, key, thres_range)

    # stop_condition, when the space is large, meet another keyword, meet a line
    list_boxes_candidate = []
    for index, item in enumerate(list_boxes_info):
        (xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color) = item
        if ((xmin < v_thres_max) and (xmax > v_thres_min)) and (ymax_k < ymin):
            list_boxes_candidate.append(
                (xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color, index))

    sorted_list_boxes_candidate = sorted(
        list_boxes_candidate, key=lambda tup: (tup[1], tup[0]))

    # key = lambda x: (x[1], x[2])

    return sorted_list_boxes_candidate


def detect_vertical_values(list_boxes_info, key, binary_img, thres_range):

    sorted_list_boxes_candidate = find_listboxes_below(
        list_boxes_info, key, thres_range=thres_range)

    if len(sorted_list_boxes_candidate) == 0:
        return

    (xmin_k, ymin_k, xmax_k, ymax_k, flag_key_k,
     textbox_content_k, box_color_k) = key

    # make vertical boxes
    for index, item in enumerate(sorted_list_boxes_candidate):
        (xmin, ymin, xmax, ymax, flag_key,
         textbox_content, box_color, index_p) = item
        img_line = binary_img[ymax_k+1:ymin-1, xmin:xmin+100]
        flag_line, _ = check_hor_line_images(img_line)
        # first item below line is value
        if index == 0:
            flag_line = False

        if (flag_key == False and flag_line == False):
            box_color = box_color_k
            list_boxes_info[index_p] = (
                xmin, ymin, xmax, ymax, flag_key, textbox_content, box_color)
            ymax_k = ymax
        else:
            break


def calculate_key_range(binary_img, key, thres_range=200):
    (xmin_k, ymin_k, xmax_k, ymax_k, flag_key_k,
     textbox_content_k, box_color_k) = key

    # if textbox_content_k == "DEL":
    #     print("debug 2")

    v_thres_min = xmin_k - thres_range
    v_thres_max = xmax_k + thres_range

    img_left = binary_img[ymin_k:ymax_k, xmin_k - thres_range:xmin_k]
    img_right = binary_img[ymin_k:ymax_k:, xmax_k:xmax_k + thres_range]

    flag_line_left, x_delta_left = check_ver_line_images(img_left)
    flag_line_right, x_delta_right = check_ver_line_images(img_right)

    v_thres_min = v_thres_min + x_delta_left
    v_thres_max = v_thres_max - (thres_range - x_delta_right)

    return v_thres_min, v_thres_max


def check_ver_line_images(image):
    h, w = image.shape
    for i in range(w-1):
        line = image[0:h, i:i+1]
        n_white_pix = np.sum(line == 255)
        if n_white_pix > 80*h/100:
            return True, i
    return False, 0


def check_hor_line_images(image):
    h, w = image.shape
    for i in range(h-1):
        line = image[i:i+1, 0:w]
        n_white_pix = np.sum(line == 255)
        if n_white_pix > 80*w/100:
            return True, i
    return False, 0


def crop_contours(img, cnt):

    x0, y0, w0, h0 = cv2.boundingRect(cnt)
    th1 = img[y0:y0+h0, x0:x0+w0]

    _, contours, _ = cv2.findContours(
        th1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_x = []
    list_y = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        list_x.append(x)
        list_x.append(x+w)
        list_y.append(y)
        list_y.append(y+h)

    x1 = min(list_x)
    y1 = min(list_y)
    x2 = max(list_x)
    y2 = max(list_y)
    return x0+x1, y0+y1, x2-x1, y2-y1

# remove line horizal and vertical


def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def remove_lines(img):

    # edges = cv2.Canny(img, 100, 200, apertureSize=3)
    edges = auto_canny(img)
    cv2.imwrite("test/canny.jpg", edges)

    # horizal
    minLineLength = 1000
    maxLineGap = 100
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1000,
                            minLineLength, maxLineGap)
    if len(lines) == 0:
        return img
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)

    # vertical
    minLineLength = 200
    maxLineGap = 100
    lines = cv2.HoughLinesP(edges, 90, (90 * np.pi) / 180, 1000,
                            minLineLength, maxLineGap)
    if len(lines) == 0:
        return img
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)

    return img

# delete and make new folder


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


if __name__ == "__main__":

    # loading images
    list_files = os.listdir("IMG_SI/")

    recreate_folder('test')
    folder_test = "test/"
    num_test = 3

    list_test_file = ['doc_22808.png', 'doc_22900.png', 'doc_6097.png',
                      'doc_30420.png', 'doc_55185.png', 'doc_30945.png', 'doc_46068.png', 'doc_57696.png']
    list_file_test = []

    # statistic keyword
    list_ocr_strings_all = []

    for i in range(num_test):
        file = random.choice(list_files)
        print(file)
    # for file in list_files:
        # if "doc_22900.png" not in file:
        #     continue

        start_time = time.time()
        try:
            image1 = cv2.imread("IMG_SI/" + file)
            output1_line = image1.copy()
        except:
            continue

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # clean the image using otsu method with the inversed binarized image
        ret1, th1 = cv2.threshold(
            gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # th1 = cv2.adaptiveThreshold(
        #     gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        cv2.imwrite(folder_test+"s1_binary.jpg", th1)
        binary_img = th1.copy()

        try:
            th1 = remove_lines(th1)
            cv2.imwrite(folder_test+"s2_no_lines.jpg", th1)
        except:
            print("except")

        # processing and writing the output
        output1_line, white_bg, image_key_value, list_ocrstring = process_line_advanced(
            th1, output1_line, binary_img)
        # special case for the 5th output because margin with paragraph is just the 4th output with margin
        file_name, file_extension = os.path.splitext(file)
        cv2.imwrite(folder_test + file_name+"_line.jpg", output1_line)
        cv2.imwrite(folder_test + file_name+"_wbg.jpg", white_bg)
        cv2.imwrite(folder_test + file_name+"_kv.jpg", image_key_value)

        list_ocr_strings_all.append(list_ocrstring)

        print(file)
        list_file_test.append(file)
        print("--- %s seconds ---" % (time.time() - start_time))

    with open('ocr_statistic.pickle', 'wb') as handle:
        pickle.dump(list_ocr_strings_all, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open('ocr_statistic.pickle', 'rb') as handle:
        list_ocr_strings_all = pickle.load(handle)

    print(list_ocr_strings_all)
    print(list_file_test)


# neu text line nao ma co dinh line thi thuc hien tach ra
# mot so field nam trong box qua day nen boxes.jpg se khong thay
# sau khi textbox, co the plot no len 1 image khac de thuc hien ocr 1 lan
# thuat toan tim key chua dung xac suat ra key tot hon
