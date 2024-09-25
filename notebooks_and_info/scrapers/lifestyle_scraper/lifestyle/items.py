# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class LifestyleItem(scrapy.Item):
    name = scrapy.Field()
    search_url = scrapy.Field()
    isolation_total = scrapy.Field()
    isolation_src = scrapy.Field() # dict
