import os
import requests
import json
from bs4 import BeautifulSoup

from const.const import ScrapingConst, HttpStatusCode, ErrorMessage


class BrandCode:
    
    @classmethod
    def get_html_info(cls, url):
        req = requests.get(url)
        req.encoding = req.apparent_encoding
        
        if req.status_code == HttpStatusCode.NOT_FOUND.value:
            raise Exception(ErrorMessage.NOT_FOUND_MSG.value)
        elif req.status_code == HttpStatusCode.TIMEOUT.value:
            raise Exception(ErrorMessage.TIMEOUT_MSG.value)

        bsObj = BeautifulSoup(req.text,"html.parser")
        
        return bsObj
    
    @classmethod
    def target_info(cls, bsObj, tag, search, attr="href"):
        target_obj = {}
        href_search = "stock_sec_code_mul="
        
        items = bsObj.find_all(tag, href=True)
        
        links = [a for a in items if search in a[attr]]

        # 抽出したリンクを表示
        for link in links:
            idx = link[attr].find(href_search)+len(href_search)
            target_obj[link.get_text()] = link[attr][idx:idx+4]
            
        return target_obj
    
    @classmethod
    def get_text(cls, dict, path=ScrapingConst.DIR.value + "/" + ScrapingConst.FILE_NAME.value):
        os.makedirs(ScrapingConst.DIR.value, exist_ok=True)
        output = path
        with open(output, 'w', encoding='utf-8') as file:
            json.dump(dict, file, ensure_ascii=False, indent=4)

        print("ファイルに書き込みました。")


def main():
    bsObj = BrandCode.get_html_info(ScrapingConst.URL.value)
    target_obj = BrandCode.target_info(bsObj, ScrapingConst.TAG.value, ScrapingConst.SEARCH.value)
    BrandCode.get_text(target_obj)

        
# if __name__ == "__main__":
#     main()
