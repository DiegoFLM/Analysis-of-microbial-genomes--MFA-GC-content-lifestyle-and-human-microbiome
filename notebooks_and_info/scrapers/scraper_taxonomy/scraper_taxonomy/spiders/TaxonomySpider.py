import scrapy
import pandas as pd
from pathlib import Path

def load_urls():
    PATH_DATA = Path.cwd().parent.parent.parent.parent / 'data'
    PATH_PREPROCESSED = PATH_DATA / 'preprocessed'
    # PATH_DF_URLS_ARCHAEA = PATH_PREPROCESSED / 'archaea' / 'archaea_name_formatting.csv'

    # df_archaea_name_formatting = pd.read_csv(PATH_DF_URLS_ARCHAEA)

    # names = df_archaea_name_formatting['name'].tolist()
    # url_prefix = 'https://www.ncbi.nlm.nih.gov/taxonomy/?term='


    PATH_DF_URLS_BACTERIA = PATH_PREPROCESSED / 'bacteria' / 'bacteria_name_formatting.csv'

    df_bacteria_name_formatting = pd.read_csv(PATH_DF_URLS_BACTERIA)

    names = df_bacteria_name_formatting['name'].tolist()
    url_prefix = 'https://www.ncbi.nlm.nih.gov/taxonomy/?term='

    urls_names = []

    for name in names:
        url = url_prefix + name
        urls_names.append((url, name))  # Store both the URL and the organism name
    return urls_names


class TaxonomyspiderSpider(scrapy.Spider):
    name = "TaxonomySpider"
    allowed_domains = ["www.ncbi.nlm.nih.gov"]
    start_urls_with_names = load_urls()

    def start_requests(self):
        for url, name in self.start_urls_with_names:
            yield scrapy.Request(url=url, callback=self.parse, meta={'organism_name': name})


    def parse(self, response):
        relative_url = response.css("div.rprt p a::attr(href)").get()
        absolute_url = response.urljoin(relative_url)
        organism_name = response.meta['organism_name']  # Retrieve organism name from meta
        yield scrapy.Request(url=absolute_url, callback=self.parse_taxonomy, meta={'organism_name': organism_name})

    def parse_taxonomy(self, response):
        taxonomic_ranks = response.css("body form table tr td dl dd a::attr(title)").getall()
        taxonomic_lineage =  response.css("body form table tr td dl dd a::text").getall()
        item = {
            'url': response.url,
            'organism_name': response.meta['organism_name']  # Include organism name in the item
        }
        for rank in taxonomic_ranks:
            item[rank] = taxonomic_lineage[taxonomic_ranks.index(rank)]
        yield item




if __name__ == "__main__":
    urls = load_urls()
    for url in urls:
        print(url)