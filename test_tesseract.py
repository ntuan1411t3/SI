import pytesseract
import cv2
import os
import shutil
import numpy as np


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


def draw_boxes(image, list_characters):

    try:
        list_characters.pop(0)
        for c in list_characters:
            char_info = c.split('\t')
            # char left bottom right top page
            text = char_info[11]
            if text == "":
                continue
            left = int(char_info[6])
            top = int(char_info[7])
            w = int(char_info[8])
            h = int(char_info[9])
            cv2.rectangle(image, (left, top), (left+w, top+h), (0, 255, 0), 1)
            cv2.putText(image, text, (left, top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
    except:
        print("except")

    return image


if __name__ == "__main__":

    recreate_folder("test_tesseract/")

    # list_files = os.listdir("IMG_SI/")
    # for file in list_files:
    #     if "doc_13506.png" not in file:
    #         continue

    #     image = cv2.imread('IMG_SI/'+file)
    #     string_characters = pytesseract.image_to_data(image, lang="eng")
    #     list_characters = string_characters.split("\n")
    #     cv2.imwrite("test_tesseract/"+file, draw_boxes(image, list_characters))
    #     print(file)

    file = 'test.png'
    image = cv2.imread(file)

    str1 = pytesseract.image_to_string(
        image, config='-l eng --tessdata-dir "tessdata" --psm 3')
    print(str1)

    str_list = str1.split("\n")
    str_list = [x for x in str_list if x]
    if " " in str_list:
        str_list.remove(" ")
    print(str_list)
    print(len(str_list))

    # file = "myc_100.jpg"
    # image = cv2.imread(file)
    # file2 = "ocr.jpg"
    # image2 = cv2.imread(file2)
    # h2, w2, _ = image2.shape
    # image = cv2.resize(image, (h2, h2))
    # vis = np.concatenate((image, image2), axis=1)
    # cv2.imwrite("vis.jpg", vis)
    # image = cv2.imread("vis.jpg")
    # print(pytesseract.image_to_string(image, lang="eng", config='--psm 13'))

    # string_characters = pytesseract.image_to_data(image, lang="eng")
    # list_characters = string_characters.split("\n")
    # cv2.imwrite("test_tesseract/test.png",
    #             draw_boxes(image.copy(), list_characters))
