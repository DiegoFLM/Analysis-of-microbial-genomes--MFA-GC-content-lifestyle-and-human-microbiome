import scrapy
import pandas as pd
from pathlib import Path

from lifestyle.items import LifestyleItem

def load_urls():
    PATH_DATA = Path.cwd().parent.parent.parent.parent.parent / 'data'
    PATH_PREPROCESSED = PATH_DATA / 'preprocessed'
    PATH_DF_URLS_ARCHAEA = PATH_PREPROCESSED / 'archaea' / 'archaea_name_formatting.csv'

    df_archaea_name_formatting = pd.read_csv(PATH_DF_URLS_ARCHAEA)
    urls = df_archaea_name_formatting['url_gcm_isolation_src'].tolist()
    return urls

class LifestylespiderSpider(scrapy.Spider):
    name = "lifestylespider"
    allowed_domains = ["https://gcm.wdcm.org/"]
    start_urls = load_urls()

    def parse(self, response):
        pass
