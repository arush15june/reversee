import os
from io import BytesIO
from random import randint
import requests
from PIL import Image

class ImageScraper():
        def __init__(self, url, rows, pages, path):
            self.url = url
            self.rows = rows
            self.pages = pages
            self.path = path

        @staticmethod
        def random_file_name(format):
            return str(randint(100000, 999999)) + '.' + format
            
        @staticmethod
        def save_image(url, name, path):
            r = requests.get(url)
            image_file = BytesIO(r.content)
            try:
                image = Image.open(image_file)
                image.save(os.path.join(path, name), quality=90)
                return r.status_code
            except:
                return None

        @staticmethod
        def get_json(url, params):
            r = requests.get(url, params=params)
            if r.status_code == 200:
                return r.json(), r.url
            else:
                return None

        def scrape(self):
            TOTAL_COUNTER = 0
            for page in range(self.pages[0], self.pages[1]):
                PAGE_COUNTER = 0

                print(f"Page: {page}")
                
                curr_params = {
                    'f' : '',
                    'p' : page,
                    'rows' : self.rows
                }
                result, url = self.get_json(self.url, curr_params)
                print(f"Scraped Url: {url}")
                
                products = result['data']['results']['products']
                print(f"Products: {len(products)}")
                for product in products:
                    image_name = self.random_file_name('jpg')
                    image_url = product['search_image']
                    status = self.save_image(image_url, image_name, self.path)
                    if status is None:
                        print(f"Failed To Save Image: {image_name} Page Count: {PAGE_COUNTER+1} Total Count: {TOTAL_COUNTER+1}")
                    else:
                        print(f"Saved Image: {image_name} Page Count: {PAGE_COUNTER+1} Total Count: {TOTAL_COUNTER+1}")
                        PAGE_COUNTER += 1
                        TOTAL_COUNTER += 1
            



if __name__ == '__main__':
    TSHIRT_URL =  os.environ.get("TSHIRT_URL") 
    IMAGES_PATH = 'images'
    scraper = ImageScraper(TSHIRT_URL, 100, (11, 100), IMAGES_PATH) # args: p => pages, rows => rows
    scraper.scrape()