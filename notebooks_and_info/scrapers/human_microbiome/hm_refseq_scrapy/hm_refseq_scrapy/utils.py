# utils.py
from pathlib import Path
import pandas as pd

# Define base paths
PATH_DATA = Path.cwd().parent.parent.parent.parent.parent / 'data'
PATH_SCRAPED = PATH_DATA / 'raw' / 'scraped'
PATH_HM = PATH_SCRAPED / 'human_microbiome'
PATH_HM_FNA = PATH_HM / 'genomes' / 'bacteria'
PATH_HM_REFSEQ_LIST = PATH_HM / 'refseq' / 'hm_refseq_list.csv'

ABS_PATH_HM_FNA = PATH_HM_FNA.resolve()

def get_hm_urls():
    df_hm_refseq_list = pd.read_csv(PATH_HM_REFSEQ_LIST)
    hm_urls = df_hm_refseq_list['url'].tolist()
    return hm_urls
