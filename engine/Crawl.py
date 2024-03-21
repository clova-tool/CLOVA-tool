import os
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from PIL import Image
import urllib3
import time
urllib3.disable_warnings()
import re

requests.adapters.DEFAULT_RETRIES = 5

def crawl_google(url, images_path, keyword, pic_num, is_face, FaceDetInterpreter, SegmentInterpreter, only_image=False,category_name=''):
    response = requests.get(url)

    # 发送HTTP请求并获取页面内容
    response = requests.get(url)

    # 使用Beautiful Soup解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有的图片标签（通常是 <img> 标签）
    img_tags = soup.find_all('img')
    # print(img_tags)
    # 创建一个目录来保存爬取的图片
    os.makedirs(os.path.join(images_path, keyword), exist_ok=True)

    # 遍历图片标签，下载图片并保存到目录中
    num = 0
    # print(len(img_tags))
    
    images_info_list = []
    for img_tag in img_tags:
        images_info = {}
        if num < pic_num:
            img_url = img_tag.get('src')  # 获取图片的链接
            if img_url.startswith("https://"):
                img_response = requests.get(img_url)
                # img_filename = os.path.join('downloaded_images', os.path.basename(img_url))
                img_filename = os.path.join(images_path, keyword, f"{num}.jpg")
                with open(img_filename, 'wb') as img_file:
                    img_file.write(img_response.content)
                # with Image.open(img_filename) as img:
                #     resized_img = img.resize((224, 224))

                # # 保存调整大小后的图像
                # resized_img.save(img_filename)  # 保存到另一个文件
                print(f"Downloading images: {img_filename}")
                if only_image == False:
                    if is_face:
                        image = Image.open(img_filename).convert('RGB')
                        face_dec= FaceDetInterpreter
                        faces = face_dec.det_face(image)
                        if len(faces) == 1:
                            images_info['index'] = num
                            images_info['label'] = keyword
                            images_info['image_path'] = img_filename
                            images_info['faces'] = faces
                            images_info_list.append(images_info)
                            num += 1    
                        
                    else:
                        image = Image.open(img_filename).convert('RGB')
                        seg= SegmentInterpreter
                        if category_name=='':
                            objs = seg.predict(image, '')
                        else:
                            objs = seg.predict(image,category_name)
                        if len(objs) > 0:
                            images_info['index'] = num
                            images_info['label'] = keyword
                            images_info['image_path'] = img_filename
                            images_info['objs'] = objs
                            images_info_list.append(images_info)
                            num += 1  
                else:
                    num += 1   

                    
    print("Download images finished!")
    return images_info_list


import requests
import os
import re
import time
from PIL import Image


def crawl_baidu(url, images_path, keyword, pic_num, is_face, FaceDetInterpreter, SegmentInterpreter, only_image=False, category_name=''):

    os.makedirs(os.path.join(images_path, keyword), exist_ok=True)

    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
    url = 'https://image.baidu.com/search/acjson?'
    num = 0
    page_num=1
    for pn in range(0, 30 * page_num, 30):

        param = {'tn': 'resultjson_com',
                # 'logid': '7603311155072595725',
                'ipn': 'rj',
                'ct': 201326592,
                'is': '',
                'fp': 'result',
                'queryWord': keyword,
                'cl': 2,
                'lm': -1,
                'ie': 'utf-8',
                'oe': 'utf-8',
                'adpicid': '',
                'st': -1,
                'z': '',
                'ic': '',
                'hd': '',
                'latest': '',
                'copyright': '',
                'word': keyword,
                's': '',
                'se': '',
                'tab': '',
                'width': '',
                'height': '',
                'face': 0,
                'istype': 2,
                'qc': '',
                'nc': '1',
                'fr': '',
                'expermode': '',
                'force': '',
                'cg': '',    # 这个参数没公开，但是不可少
                'pn': pn,    # 显示：30-60-90
                'rn': '30',  # 每页显示 30 条
                'gsm': '1e',
                '1618827096642': ''
                }

        while(1):
            try:
                request = requests.get(url=url, headers=header, params=param)
                break
            except:
                time.sleep(3)
                continue
        if request.status_code == 200:
            print('Request success.')
        request.encoding = 'utf-8'
        # 正则方式提取图片链接
        html = request.text
        # print('html',html)
        image_url_list = re.findall('"thumbURL":"(.*?)",', html, re.S)

        images_info_list = []
        for image_url in image_url_list:
            images_info = {}
            while(1):
                try:
                    image_data = requests.get(url=image_url, headers=header).content
                    break
                except:
                    time.sleep(3)
                    continue
            img_filename = os.path.join(images_path, keyword, f"{num}.jpg")
            with open(img_filename, 'wb') as fp:
                fp.write(image_data)

            # # 保存调整大小后的图像
            # resized_img.save(img_filename)  # 保存到另一个文件
            print(f"Downloading images: {img_filename}")

            if only_image == False:
                if is_face:
                    image = Image.open(img_filename).convert('RGB')
                    face_dec= FaceDetInterpreter
                    faces = face_dec.det_face(image)
                    if len(faces) == 1:
                        images_info['index'] = num
                        images_info['label'] = keyword
                        images_info['image_path'] = img_filename
                        images_info['faces'] = faces
                        images_info_list.append(images_info)
                        num += 1    
                    
                else:
                    image = Image.open(img_filename).convert('RGB')
                    seg= SegmentInterpreter
                    if category_name=='':
                        objs = seg.predict(image, '')
                    else:
                        objs = seg.predict(image,category_name)
                    if len(objs) > 0:
                        images_info['index'] = num
                        images_info['label'] = keyword
                        images_info['image_path'] = img_filename
                        images_info['objs'] = objs
                        images_info_list.append(images_info)
                        num += 1  
            else:
                num += 1   

            if num >= pic_num:
                break

    print("Download images finished!")
    return images_info_list













