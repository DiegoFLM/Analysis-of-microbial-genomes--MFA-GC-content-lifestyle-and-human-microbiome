from gcm.gcm import GCM
from gcm import constants as const


with GCM(teardown = True) as bot:
    bot.scrape_data(
        PATH_DF_NAME_FORMATTING = const.PATH_DF_URLS_BACTERIA_NO_DUPS, 
        PATH_DESTINY = const.PATH_DESTINY_LIFESTYLE_BACTERIA,
        column = 3
    )
    print("Exiting...")


