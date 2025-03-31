import requests
import json
import os
from lxml import etree
import time

# 目标链接（需解码URL参数）
url = "https://www.bing.com/images/search?q=%e8%bf%aa%e4%b8%bd%e7%83%ad%e5%b7%b4%e5%9b%be%e7%89%87&qpvt=%e8%bf%aa%e4%b8%bd%e7%83%ad%e5%b7%b4%e5%9b%be%e7%89%87&form=IGRE&first={page}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

# 创建保存目录
save_dir = "E:/桌面/CV/images/face/dilireba/"
os.makedirs(save_dir, exist_ok=True)

def download_images(start_page=1, max_pages=10):
    page = start_page
    count = 298
    while page <= max_pages:
        try:
            # 构造分页URL（每页35张）
            target_url = url.format(page=page)
            response = requests.get(target_url, headers=headers)
            response.encoding = "utf-8"
            html = etree.HTML(response.text)

            # 解析图片JSON数据（必应异步加载的图片信息存储在a标签的m属性）
            img_tags = html.xpath('//a[@class="iusc"]/@m')
            for tag in img_tags:
                img_data = json.loads(tag)
                img_url = img_data.get("murl")  # 提取原图链接
                if not img_url:
                    continue

                # 下载图片
                try:
                    img_response = requests.get(img_url, headers=headers, timeout=10)
                    if img_response.status_code == 200:
                        with open(os.path.join(save_dir, f"{count}.jpg"), "wb") as f:
                            f.write(img_response.content)
                        print(f"下载成功：第{count}张")
                        count += 1
                    time.sleep(0.5)  # 降低请求频率
                except Exception as e:
                    print(f"下载失败：{img_url}，错误：{e}")

            # 翻页（必应每页固定35张）
            page += 35
            time.sleep(1)  # 每页间隔1秒

        except Exception as e:
            print(f"页面解析异常：{e}")
            break

if __name__ == "__main__":
    download_images(start_page=301, max_pages=300)  # 示例爬取前5页（约175张）