import os
import cv2

json_root = 'fake/labels'
image_root = 'fake/images'
json_list = os.listdir(json_root)
for n in json_list:
    image_path = os.path.join(image_root, n.split('.')[0] + '.jpg')
    json_path = os.path.join(json_root, n)
    print('>>>', image_path, json_path)
    image = cv2.imread(image_path)
    try:
        h, w = image.shape[:2]
    except Exception as e:
        print(e)
    for i in open(json_path):
        data = i[:-1]
        data_split = data.split(' ')
        first_point = (int((float(data_split[1]) - float(data_split[3]) / 2) * w),
                       int((float(data_split[2]) - float(data_split[4]) / 2) * h))
        last_point = (int((float(data_split[1]) + float(data_split[3]) / 2) * w),
                      int((float(data_split[2]) + float(data_split[4]) / 2) * h))
        cv2.rectangle(image, first_point, last_point, (0, 255, 0), 2)
    cv2.imwrite(image_path + '.jpg', image)
