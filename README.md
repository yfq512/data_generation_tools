<h2 align="center">
Useful tools for computer vision/deep learning.
</h2>
<h4 align="center">
    <p><b>简体中文</b> | <a href="https://github.com/yfq512/data_generation_tools/blob/main/README_EN.md">English</a><p>
</h4>

### 数据增强
* 针对图像随机仿射、旋转、平移、拉伸、颜色噪声、亮度等方面随机数据增强
* 默认仅针对目标检测数据增强，生成图像及对应的yolo格式标注文件（图像分类需对代码就行部分修改）
* 你可以指定生成数据的数量 N，但实际生成数据的数量约为 0.8*N

### 依赖
* pip install opencv-python -i https://mirror.baidu.com/pypi/simple

### 使用步骤
* 基于原始图像(data/imgs)和分类标签(data/labels.txt)用在线标注工具 https://www.makesense.ai/ 制作vgg格式的标注文件，也可以导入coco标注文件再导出vgg格式的标注文件 (data/vgg.json 此处文件已存在，用户也可重新制作)
  ![image](https://github.com/yfq512/data_generation_tools/blob/main/imgs/1.jpg)
  ![image](https://github.com/yfq512/data_generation_tools/blob/main/imgs/2.jpg)
* 生成数据 python generator.py 生成的图像数据存放在(fake/images)，对应标签存放在(fake/labels)
* 验证生成数据 python plot.py 将标注文件的box画到对应的生成图像，画好图像保存在(fake/images)

### 其他
* 若需要将yolo格式转成coco格式，请参考 https://github.com/yfq512/DL_tools
