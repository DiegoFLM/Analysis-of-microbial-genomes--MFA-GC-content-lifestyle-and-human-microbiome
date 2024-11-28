# import scrapy
# from scrapy.pipelines.files import FilesPipeline
# from ..items import HmRefseqScrapyItem
# from ..utils import get_hm_urls

# from urllib.parse import urljoin
# import pandas as pd
# from pathlib import Path


# class HmRefseqSpider(scrapy.Spider):
#     name = "hm_refseq"
#     allowed_domains = ["ftp.ncbi.nlm.nih.gov"]
#     start_urls = ["https://ftp.ncbi.nlm.nih.gov/genomes/HUMAN_MICROBIOM/Bacteria/Acinetobacter_johnsonii_SH046_uid38339/"]

#     def parse(self, response):
#         name = (response.url).split("/")[-1]
#         resources_urls = response.xpath('//pre/a/@href').getall()
#         for resource_url in resources_urls:
#             if resource_url.endswith('.fna.tgz'):
#                 fna_url = urljoin(response.url, resource_url)
#                 genome_file_name = f"{name}/{resource_url}"
#                 item = HmRefseqScrapyItem()
#                 item['file_urls'] = [fna_url]
#                 item['genome_file_name'] = genome_file_name
#                 yield item




import scrapy
from urllib.parse import urljoin
from ..items import HmRefseqScrapyItem
from ..utils import get_hm_urls

class HmRefseqSpider(scrapy.Spider):
    name = "hm_refseq"
    allowed_domains = ["ftp.ncbi.nlm.nih.gov"]
    # start_urls = [
    #     "https://ftp.ncbi.nlm.nih.gov/genomes/HUMAN_MICROBIOM/Bacteria/Acinetobacter_johnsonii_SH046_uid38339/"
    # ]
    start_urls = get_hm_urls()

    def parse(self, response):
        name = response.url.rstrip('/').split("/")[-1]
        resources_urls = response.xpath('//pre/a/@href').getall()
        for resource_url in resources_urls:
            if resource_url.endswith('.fna.tgz'):
                fna_url = urljoin(response.url, resource_url)
                # Use forward slashes to define the path
                genome_file_name = f"{name}/{resource_url}"
                item = HmRefseqScrapyItem()
                item['file_urls'] = [fna_url]
                item['genome_file_name'] = genome_file_name
                yield item
