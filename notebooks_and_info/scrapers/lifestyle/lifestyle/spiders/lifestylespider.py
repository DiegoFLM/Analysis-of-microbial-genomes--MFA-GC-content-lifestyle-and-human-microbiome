import scrapy


class LifestylespiderSpider(scrapy.Spider):
    name = "lifestylespider"
    allowed_domains = ["www.ncbi.nlm.nih.gov"]
    start_urls = ["https://www.ncbi.nlm.nih.gov/nuccore/"]

    def parse(self, response):
        pass
