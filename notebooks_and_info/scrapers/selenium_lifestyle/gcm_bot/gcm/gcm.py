import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import time
from pathlib import Path
import pandas as pd
import json

import gcm.constants as const


class GCM:
    def __init__(self, driver_path=r"/usr/local/bin", teardown=False):
        self.teardown = teardown
        self.driver_path = driver_path
        os.environ['PATH'] += f":{self.driver_path}"

        # with Firefox
        options = webdriver.FirefoxOptions()
        if teardown:
            pass
        else:
            # options.set_preference("detach", True) 
            options.log.level = "trace"

        options.headless = True
        self.driver = webdriver.Firefox(options=options)


        self.driver.implicitly_wait(50)
        self.driver.maximize_window()


    def __enter__(self):
        return self  # Returning 'self' so it can be used within the 'with' block

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.teardown:
            self.driver.quit()


    def load_df_name_formatting(self, PATH_DF_NAME_FORMATTING):
        df_name_formatting = pd.read_csv(PATH_DF_NAME_FORMATTING)
        return df_name_formatting
    
    def scrape_data(self, PATH_DF_NAME_FORMATTING, PATH_DESTINY, column):
        df_name_formatting = self.load_df_name_formatting(PATH_DF_NAME_FORMATTING = PATH_DF_NAME_FORMATTING)
        # dict_list = []
        with open(PATH_DESTINY, 'w') as f:
            f.write("[\n")

        existing_dicts = False
        with open(PATH_DESTINY, 'a') as f:
            for index, row in df_name_formatting.loc[:].iterrows():
                # print(index, row[['name', 'url_gcm_isolation_src']])
                self.driver.get(row['url_gcm_isolation_src'])
                table_rows = self.driver.find_elements("css selector", "div.fd-isolate table tr")
                found = False
                dict_organism = {}
                dict_organism['name'] = df_name_formatting.loc[index, 'name']
                dict_organism['url_refseq'] = df_name_formatting.loc[index, 'url']
                dict_organism['url_gcm_isolation_src'] = df_name_formatting.loc[index, 'url_gcm_isolation_src']
                
                for row in table_rows[1:]:
                    environment = row.text.split("\n")[0]
                    val = int(str(row.text.split("\n")[column]).replace(",", ""))
                    if val > 0:
                        found = True
                        dict_organism[environment] = val
                    
                if found:
                    # dict_list.append(dict_organism)
                    if existing_dicts:
                        f.write(",\n")
                    json.dump(dict_organism, f, indent=2)
                    existing_dicts = True
                print(dict_organism)


        with open(PATH_DESTINY, 'a') as f:
            f.write("\n]")



