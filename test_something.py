import cv2
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny

# def crop_text_region(image, cnt, index):
#     x, y, w, h = cv2.boundingRect(cnt)
#     img = image[y:y+h, x:x+w]
#     # extract green color
#     # convert to hsv
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # mask of green (36,0,0) ~ (70, 255,255)
#     mask = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))
#     # slice the green
#     imask = mask > 0
#     green = np.zeros_like(img, np.uint8)
#     green[imask] = img[imask]
#     cv2.imwrite("green.jpg", green)

#     # binary and crop image
#     gray1 = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
#     ret1, th1 = cv2.threshold(
#         gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     x1, y1, x2, y2 = find_extreme_points(th1)
#     return x + x1, y + y1, x2 - x1, y2 - y1

# # get foreground text
# def get_foreground_text(crop_img):
#     crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#     ret, crop_img = cv2.threshold(
#         crop_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     n_black_pix = np.sum(crop_img == 0)
#     n_white_pix = np.sum(crop_img == 255)
#     if n_white_pix < n_black_pix:
#         ret, crop_img = cv2.threshold(
#             crop_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#     crop_img = np.expand_dims(crop_img, axis=2)
#     return crop_img

# # processing line advanced by line boxing ( fast version )


# def process_line_advanced_fast(thresh, output, binary_img):

#     thresh, _ = take_character_boxes(thresh)
#     cv2.imwrite("test/boxes.jpg", thresh)
#     kernel2 = np.ones((1, 60), np.uint8)
#     line_img = cv2.dilate(thresh, kernel2, iterations=1)
#     cv2.imwrite("test/line_pre.jpg", line_img)

#     _, contours, _ = cv2.findContours(
#         line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     img_kv = output.copy()
#     output2 = output.copy()
#     height, width, _ = output.shape
#     # make backround white
#     white_bg = np.ones((height, width, 3), np.uint8) * 255
#     color_red = (0, 0, 255)
#     color_blue = (255, 0, 0)
#     color_green = (0, 255, 0)
#     color_black = (0, 0, 0)

#     # dict keyword
#     # dict_key_unique = {}
#     # list of boxes info
#     list_boxes_info = []

#     # get all boxes
#     list_all_lines = []
#     total_h = 0
#     for index, cnt in enumerate(contours):
#         # try crop text region
#         x, y, w, h = crop_contours(thresh, cnt)
#         if w < 10 or h < 10 or h > 100:
#             continue
#         list_all_lines.append((x, y, w, h))
#         total_h += (h + 30)

#     # check max w and total h
#     max_w = sorted(list_all_lines, key=lambda tup: tup[2], reverse=True)[0][2]
#     max_w += 50
#     # make a big picture for all lines
#     big_bg = np.ones((total_h, max_w, 3), np.uint8) * 255
#     hh = 0
#     for index, item in enumerate(list_all_lines):
#         (x, y, w, h) = item
#         # ocr text box
#         crop_img = output2[y - 3:y + h + 3, x - 3:x + w + 3]
#         hc, wc, _ = crop_img.shape
#         # get foreground of crop img (test)
#         crop_img = get_foreground_text(crop_img)
#         big_bg[hh:hh+hc, 0:0+wc] = crop_img
#         # cv2.line(big_bg, (0, hh + hc + 2), (max_w, hh + hc + 2), (0, 0, 0), 2)
#         hh += hc + 20

#     print(len(list_all_lines))
#     # save test
#     cv2.imwrite("test.png", big_bg)

#     return None, None, None

#     ocr_strings = image_to_string(
#         big_bg, config='-l eng --tessdata-dir "tessdata" --psm 13').split("\n")
#     ocr_strings = [x for x in ocr_strings if x]

#     if " " in ocr_strings:
#         ocr_strings.remove(" ")
#     # print(len(list_all_lines))
#     # print(len(ocr_strings))
#     list_boxes_ocr = []
#     for index, item in enumerate(ocr_strings):
#         (x, y, w, h) = list_all_lines[index]
#         list_boxes_ocr.append((x, y, w, h, item))

#     for index, box_ocr in enumerate(list_boxes_ocr):
#         (x, y, w, h, ocr_result) = box_ocr
#         # flag key or not
#         flag_key = False
#         textbox_content = item

#         # detect key or not
#         try:
#             keyword = keyword_detection(ocr_result)
#             label = keyword[0]
#             percent = keyword[1]
#             if label != None:
#                 # if label not in dict_key_unique:
#                 #     dict_key_unique[label] = (percent, x, y, w, h)
#                 # else:
#                 #     if percent < dict_key_unique[label][0]:
#                 #         dict_key_unique[label] = (percent, x, y, w, h)
#                 cv2.putText(output, str(label) + " : " + str(percent), (x, y-3),
#                             cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color_blue, thickness=2)
#                 cv2.putText(white_bg, str(label), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=1, color=color_black, thickness=2)
#                 flag_key = True
#                 textbox_content = label

#         except:
#             print("can not detect key")
#         print(str(index) + " : " + ocr_result)
#         # append box to list boxes
#         box_color = (0, 0, 0)
#         if flag_key == True:
#             box_color = name_to_rgb(dict_label_color[textbox_content])

#         list_boxes_info.append(
#             (x, y, x+w, y+h, flag_key, textbox_content, box_color))

#         cv2.rectangle(white_bg, (x, y), (x + w, y + h),
#                       color=color_black, thickness=2)
#         cv2.rectangle(output, (x - 3, y - 3),
#                       (x + w + 3, y + h + 3), color=color_red, thickness=2)
#         cv2.putText(output, str(index), (x+w, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color_red, thickness=2)

#     # make key value image
#     image_key_value = make_key_value(binary_img, list_boxes_info, output2)

#     opacity = 0.5
#     cv2.addWeighted(image_key_value, opacity, img_kv,
#                     1 - opacity, 0, img_kv)

#     return output, white_bg, img_kv

def check_line_images(image):
    h, w = image.shape
    for i in range(h-1):
        line = image[i:i+1, 0:w]
        n_black_pix = np.sum(line == 255)
        if n_black_pix > 50*w/100:
            return True
    return False


def main():
    pass


if __name__ == '__main__':
    main()
    image = cv2.imread("IMG_SI/doc_16681.png")
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(
        gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = canny(th1, 2, 1, 25)

    lines = probabilistic_hough_line(edges, threshold=100, line_length=1000,
                                     line_gap=50)

    for line in lines:
        p0, p1 = line
        cv2.line(th1, (p0[0], p0[1]), (p1[0], p1[1]), (0, 0, 0), 1)

    cv2.imwrite("test.png", th1)

    # crop_img = th1[384+1:410-1, 1314:1314+100]
    # cv2.imwrite("aaa.png", crop_img)
    # print(check_line_images(crop_img))
    # cv2.rectangle(th1, (1314, 384+1), (1314+100, 410-1),
    #               color=(255, 255, 255), thickness=cv2.FILLED)
    # cv2.imwrite("bbb.png", th1)

    # image = cv2.imread("myc2.jpg")
    # image = cv2.resize(image, (100, 100))
    # cv2.imwrite("myc_100.jpg", image)
