from pathlib import Path

PATH_DATA = Path.cwd().parent.parent.parent.parent / 'data'
PATH_PREPROCESSED = PATH_DATA / 'preprocessed'
PATH_DF_URLS_ARCHAEA = PATH_PREPROCESSED / 'archaea' / 'archaea_name_formatting.csv'
PATH_RESULTS = PATH_DATA / 'results'
PATH_RESULTS_LIFESTYLE_ARCHAEA = PATH_RESULTS / 'lifestyle' / 'archaea'
PATH_DESTINY_LIFESTYLE_ARCHAEA = PATH_RESULTS_LIFESTYLE_ARCHAEA / 'lifestyle_archaea.json'
# PATH_DF_URLS_BACTERIA = PATH_PREPROCESSED / 'bacteria' / 'bacteria_name_formatting.csv'
PATH_DF_URLS_BACTERIA_NO_DUPS = PATH_PREPROCESSED / 'bacteria' / 'bacteria_name_formatting_no_duplicates.csv'
PATH_RESULTS_LIFESTYLE_BACTERIA = PATH_RESULTS / 'lifestyle' / 'bacteria'
PATH_DESTINY_LIFESTYLE_BACTERIA = PATH_RESULTS_LIFESTYLE_BACTERIA / 'lifestyle_bacteria.json'