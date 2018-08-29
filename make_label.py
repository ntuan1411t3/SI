import pandas as pd
import os
import numpy as np
from bs4 import BeautifulSoup
import cv2
from webcolors import name_to_rgb
import numpy as np
import shutil
import pickle as pkl
dict_label_color = {'SHIPPER': "blue", 'CONSIGNEE': "green", 'NOTIFY': "red", 'ALSO_NOTIFY': "magenta", 'POR': "yellow", 'POL': "cyan",
                    'POD': "navy", 'DEL': "pink", 'DESCRIPTION': "purple", 'VESSEL': "gray", 'Gross Weight': "lavender", 'Measurement': "orange"}


def xml_to_beautiful(xml_file):
    xml = ""
    with open(xml_file) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)


def _load_data(folder_xml):
    list_files = os.listdir(folder_xml)
    data = []
    count = 0

    list_label_vectors = []
    list_image_vectors = []

    for index, item in enumerate(list_files):
        if ".xml" not in item:
            continue

        img_path = item.replace("line.xml", "wbg.jpg")
        img_x = cv2.imread(folder_xml+img_path)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
        img_label = img_x.copy()
        h, w = img_x.shape
        # img_label = np.zeros((h, w, 3), np.uint8)
        dict_label_region = {'SHIPPER': (0, 0, 0, 0), 'CONSIGNEE': (0, 0, 0, 0), 'NOTIFY': (0, 0, 0, 0), 'ALSO_NOTIFY': (0, 0, 0, 0), 'POR': (0, 0, 0, 0), 'POL': (0, 0, 0, 0),
                             'POD': (0, 0, 0, 0), 'DEL': (0, 0, 0, 0), 'DESCRIPTION': (0, 0, 0, 0), 'VESSEL': (0, 0, 0, 0), 'Gross Weight': (0, 0, 0, 0), 'Measurement': (0, 0, 0, 0)}

        anno = xml_to_beautiful(folder_xml+item)
        objs = anno.findAll('object')
        for obj in objs:
            obj_names = obj.findChildren('name')[0].contents[0]
            fname = anno.findChild('filename').contents[0]
            bbox = obj.findChildren('bndbox')[0]
            xmin = int(bbox.findChildren('xmin')[0].contents[0])
            ymin = int(bbox.findChildren('ymin')[0].contents[0])
            xmax = int(bbox.findChildren('xmax')[0].contents[0])
            ymax = int(bbox.findChildren('ymax')[0].contents[0])
            # color=name_to_rgb(dict_label_color[obj_names])
            # thickness = cv2.FILLED
            cv2.rectangle(img_label, (xmin, ymin),
                          (xmax, ymax), thickness=5, color=(0, 0, 0))
            data.append([fname, xmin, ymin, xmax, ymax, obj_names])
            # make label vector
            dict_label_region[obj_names] = (xmin/w, ymin/h, xmax/w, ymax/h)

        # filename_save = item.replace("_line.xml", ".png")
        cv2.imwrite("DocSI/train/image/"+str(count)+".png", img_x)

        img_x = cv2.resize(img_x, (512, 512))
        img_x = np.expand_dims(img_x, axis=2)
        list_image_vectors.append(1 - img_x / 255)
        # filename_save = item.replace("_line.xml", ".png")
        # img_label = cv2.resize(img_label, (1024, 1024))
        cv2.imwrite("DocSI/train/label/"+str(count)+".png", img_label)
        count += 1
        list_vector = []
        for k, v in dict_label_region.items():
            (xmin, ymin, xmax, ymax) = v
            list_vector.append(xmin)
            list_vector.append(xmax)
            list_vector.append(ymin)
            list_vector.append(ymax)
        list_label_vectors.append(list_vector)

    arr_labels = np.array(list_label_vectors)
    arr_imgs = np.array(list_image_vectors)
    with open("DocSI/labels_docsi.pkl", "wb") as f:
        pkl.dump(arr_labels, f)

    with open("DocSI/imgs_docsi.pkl", "wb") as f:
        pkl.dump(arr_imgs, f)

    df = pd.DataFrame(
        data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    df.to_csv("annotation")
    return df


def main():
    pass


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


if __name__ == '__main__':
    main()
    recreate_folder("DocSI")
    recreate_folder("DocSI/test")
    recreate_folder("DocSI/train/image")
    recreate_folder("DocSI/train/label")

    df = _load_data("images_si2/")
    print(df)
