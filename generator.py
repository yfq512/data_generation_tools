import os
import time
import json
import cv2
import numpy as np
import random
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor


def fun_dot(temp, k=0.2):
    return random.uniform(1 - k, 1 + k) * temp


def fun_aff(point, M, type=None):
    if type == 'm_warpPerspective_x':
        return int(
            (point[0] * M[0][0] + point[1] * M[0][1] + M[0][2]) / (point[0] * M[2][0] + point[1] * M[2][1] + M[2][2]))
    elif type == 'm_warpPerspective_y':
        return int(
            (point[0] * M[1][0] + point[1] * M[1][1] + M[1][2]) / (point[0] * M[2][0] + point[1] * M[2][1] + M[2][2]))
    elif type == 'm_warpAffine_x':
        return int(point[0] * M[0][0] + point[1] * M[0][1] + M[0][2])
    elif type == 'm_warpAffine_y':
        return int(point[0] * M[1][0] + point[1] * M[1][1] + M[1][2])
    else:
        return None


def generator(image_root, label_root, image_info_i, types=[1, 2, 3]):  # 默认不带随机翻转
    image_path = image_info_i.get('image_path')
    # print('image_path', image_path)
    image_infos = image_info_i.get('image_info')  # type=list

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 随机翻转
    if 0 in types:
        k = random.uniform(0, 1)
        if k > 0.85:  # 对角线翻转
            image = cv2.flip(image, -1)
        elif k < 0.15:  # 上下翻转
            image = cv2.flip(image, 0)
        elif k > 0.5:  # 左右翻转
            image = cv2.flip(image, 1)
        else:  # 不翻转
            pass

    # 随机仿射 + 随机拉伸（改变原图像尺寸）
    if 2 in types:
        point1 = np.array([[int(w / 3), int(h / 3)], [int(w * 2 / 3), int(h / 3)], [int(w / 3), int(h * 2 / 3)],
                           [int(w * 2 / 3), int(h * 2 / 3)]], dtype="float32")
        point2 = np.array([[int(fun_dot(w / 3, 0.1)), int(fun_dot(h / 3, 0.1))],
                           [int(fun_dot(w * 2 / 3, 0.1)), int(fun_dot(h / 3, 0.1))],
                           [int(fun_dot(w / 3, 0.1)), int(fun_dot(h * 2 / 3, 0.1))],
                           [int(fun_dot(w * 2 / 3, 0.1)), int(fun_dot(h * 2 / 3, 0.1))]], dtype="float32")
        M = cv2.getPerspectiveTransform(point1, point2)

        image = cv2.warpPerspective(image, M, (int(fun_dot(w)), int(fun_dot(h))))
        h, w = image.shape[:2]  # 图像尺寸发生变化，重新计算w，h

        del_list = []
        for i in range(len(image_infos)):
            all_points_x = image_infos[i]['all_points_x'].copy()
            all_points_y = image_infos[i]['all_points_y'].copy()
            _temp_xs = []
            _temp_ys = []
            for j in range(len(all_points_x)):
                temp_x = fun_aff((all_points_x[j], all_points_y[j]), M, type='m_warpPerspective_x')
                _temp_xs.append(temp_x)
                temp_x = max(0, min(w, temp_x))
                temp_y = fun_aff((all_points_x[j], all_points_y[j]), M, type='m_warpPerspective_y')
                _temp_ys.append(temp_y)
                temp_y = max(0, min(h, temp_y))
                image_infos[i]['all_points_x'][j] = temp_x
                image_infos[i]['all_points_y'][j] = temp_y
            _temp_xs_mean = np.mean(_temp_xs)
            _temp_ys_mean = np.mean(_temp_ys)

            if _temp_xs_mean < 0 or _temp_xs_mean > w or _temp_ys_mean < 0 or _temp_ys_mean > h:
                del_list.append(i)

        del_list = del_list[::-1]
        for i in del_list:
            # print('del ', i)
            image_infos.pop(i)

    # 随机旋转 + 随机平移
    if 1 in types:
        rot_k = int(random.uniform(0, 359))

        M2 = cv2.getRotationMatrix2D((w / fun_dot(2), h / fun_dot(2)), rot_k, 1)

        image = cv2.warpAffine(image, M2, (w, h))

        for i in range(len(image_infos)):
            all_points_x = image_infos[i]['all_points_x'].copy()
            all_points_y = image_infos[i]['all_points_y'].copy()
            for j in range(len(all_points_x)):
                image_infos[i]['all_points_x'][j] = fun_aff((all_points_x[j], all_points_y[j]), M2,
                                                            type='m_warpAffine_x')
                image_infos[i]['all_points_y'][j] = fun_aff((all_points_x[j], all_points_y[j]), M2,
                                                            type='m_warpAffine_y')

    # 随机颜色噪声 + 随机亮度调节
    if 3 in types:
        rand_x = np.random.randint(8.5, 12.5, image.shape) / 10 * random.uniform(0.7, 1.3)
        image = image * rand_x
    rand_name = str(uuid.uuid1())
    # print(rand_name)
    image_path = os.path.join(image_root, rand_name + '.jpg')

    # 制作yolo txt 标注文件
    txt_path = os.path.join(label_root, rand_name + '.txt')
    h, w = image.shape[:2]
    cntt = 0
    with open(txt_path, 'w') as f:

        for n in image_infos:  # n = {'all_points_x': all_points_x, 'all_points_y': all_points_y, 'label_id': label_index}
            all_points_x = n.get('all_points_x')
            all_points_y = n.get('all_points_y')
            label_id = n.get('label_id')
            sign, c_x, c_y, w_x, h_y = fun_getyolobox(all_points_x, all_points_y, w, h, image)
            if sign:
                f.write(str(label_id) + ' ' + str(c_x) + ' ' + str(c_y) + ' ' + str(w_x) + ' ' + str(h_y) + '\n')
                cntt += 1
    if cntt == 0:
        try:
            os.remove(txt_path)
        except Exception as e:
            print(e)
    else:
        cv2.imwrite(image_path, image)

    return image


def fun_getyolobox(all_points_x, all_points_y, w_org, h_org, image):
    x_min = min(all_points_x)
    x_max = max(all_points_x)
    y_min = min(all_points_y)
    y_max = max(all_points_y)

    area1 = (x_max - x_min) * (y_max - y_min)
    if area1 == 0:
        # print(all_points_x, all_points_y, w_org, h_org)
        return False, None, None, None, None

    x_min = max(0, min(all_points_x))
    x_max = min(w_org, max(all_points_x))
    y_min = max(0, min(all_points_y))
    y_max = min(h_org, max(all_points_y))
    area2 = (x_max - x_min) * (y_max - y_min)

    c_x = (x_min + x_max) / 2
    c_y = (y_min + y_max) / 2
    w = (x_max - x_min)
    h = (y_max - y_min)
    if area2 / area1 > 0.8:
        return True, round(c_x / w_org, 6), round(c_y / h_org, 6), round(w / w_org, 6), round(h / h_org, 6)
    else:
        # print(all_points_x, all_points_y, w_org, h_org, area2, area1, round(area2 / area1, 2))
        return False, None, None, None, None


# 输入分别为：原始图像目录，vgg格式标注文件，图像分类，生成数据目录，期望图像数量（部分生成的图像不合要求会被丢弃）
def main(image_root, vgg_json_path, class_name_path, fake_root, image_num):
    image_generatpr_root = os.path.join(fake_root, 'images')
    try:
        os.makedirs(image_generatpr_root)
    except Exception as e:
        print(e)
    label_root = os.path.join(fake_root, 'labels')  # yolo格式标注文件保存目录
    try:
        os.makedirs(label_root)
    except Exception as e:
        print(e)
    label_list = []
    for n in open(class_name_path):
        if len(n) > 0:
            label_list.append(n[:-1])
        else:
            break
    print('labels: ', label_list)
    # 读取 voc_json_path 得到 dic_data= [{'image_path':'', 'data':['polygon':[坐标点], 'label':'']}]
    # 计算小遍历次数
    dic_data = []
    json_data = json.load(open(vgg_json_path, 'r'))
    json_data_key_list = list(json_data.keys())
    for json_data_key in json_data_key_list:
        data = json_data[json_data_key].get('regions')
        data_key_list = list(data.keys())
        image_info = []
        for data_key in data_key_list:
            data2 = data[data_key]
            label_name = data2.get('region_attributes').get('label')
            label_index = label_list.index(label_name)
            all_points_x = data2.get('shape_attributes').get('all_points_x')
            all_points_y = data2.get('shape_attributes').get('all_points_y')
            image_info.append({'all_points_x': all_points_x, 'all_points_y': all_points_y, 'label_id': label_index})
        dic_data.append({'image_path': os.path.join(image_root, json_data_key), 'image_info': image_info})

    image_per_num = int(image_num / len(dic_data))
    cnt = len(dic_data)

    max_workers = 10  # 最大线程数
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Thread')
    task_list = []

    for image_info_i in dic_data:
        for i in range(image_per_num):
            while True:
                for _i, _n in enumerate(task_list):
                    if _n.done():
                        task_list.pop(_i)
                if len(task_list) < int(max_workers * 0.9):
                    task_list.append(pool.submit(generator, image_generatpr_root, label_root, deepcopy(image_info_i)))
                    print('进度：{}: {}/{}'.format(image_root, image_num, cnt))
                    cnt += 1
                    break
                else:
                    # print('waiting...')
                    time.sleep(1)

    while True:
        for _i, _n in enumerate(task_list):
            if _n.done():
                task_list.pop(_i)
        if len(task_list) == 0:
            break
        else:
            # print('waiting...')
            time.sleep(1)
            print('len task_list', len(task_list))


if __name__ == '__main__':
    main('data/imgs', 'data/vgg.json', 'data/labels.txt', 'fake', 100)
