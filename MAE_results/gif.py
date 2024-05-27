import os
import re
import imageio

# 图片所在的目录
image_directory = '/Users/tobiaspeihengli/Downloads/MAE_results/good_final'
# 输出 GIF 的文件名
output_gif = 'output4.gif'

# 定义一个函数，用于从文件名中提取数字
def extract_number(filename):
    s = re.findall(r'\d+', filename)
    return int(s[0]) if s else -1

# 获取所有 JPEG 文件，并确保只处理 .jpg 文件
images = [img for img in os.listdir(image_directory) if img.endswith(".jpg")]
# 使用提取的数字对图片文件进行排序
images.sort(key=extract_number)

# 读取图片
frames = [imageio.imread(os.path.join(image_directory, image)) for image in images]

# 将帧写入新的 GIF 文件，并设置无限循环
imageio.mimsave(os.path.join(image_directory, output_gif), frames, duration=0.1, loop=0)  # loop=0 使 GIF 无限循环

