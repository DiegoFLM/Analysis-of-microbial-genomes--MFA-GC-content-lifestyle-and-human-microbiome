# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# import scrapy
# from scrapy.pipelines.files import FilesPipeline
# import os

# class HmRefseqScrapyPipeline(FilesPipeline):
#     def get_media_requests(self, item, info):
#         return [scrapy.Request(
#             x, meta={'genome_file_name': item['genome_file_name']}
#         ) for x in item.get('file_urls', [])]


    # def file_path(self, request, response=None, info=None, *, item=None):
    #     path = request.meta['genome_file_name']
    #     # Create directory if it doesn't exist
    #     full_path = os.path.join(self.store.basedir, path)
    #     directory = os.path.dirname(full_path)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     return path
    
    # def file_path(self, request, response=None, info=None, *, item=None):
    #     # Simply return the genome_file_name; FilesPipeline will handle directories
    #     return request.meta['genome_file_name']




import scrapy
from scrapy.pipelines.files import FilesPipeline
import os
import logging

class HmRefseqScrapyPipeline(FilesPipeline):
    def file_path(self, request, response=None, info=None, *, item=None):
        # Get the genome_file_name from request meta
        path = request.meta.get('genome_file_name', '')
        # Normalize the path to use forward slashes
        path = path.replace('\\', '/')
        logging.debug(f"Saving file to: {path}")
        return path  # Return the path as-is without sanitization

    def get_media_requests(self, item, info):
        return [scrapy.Request(
            x, meta={'genome_file_name': item['genome_file_name']}
        ) for x in item.get('file_urls', [])]
