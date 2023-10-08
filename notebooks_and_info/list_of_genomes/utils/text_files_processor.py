import pandas as pd

class TextFilesProcessor:


    def get_pdseries_from_text(self, file_name, text):
        if (file_name == "annotation_hashes.txt"):
            columns_annotation_hashes = text.strip().split('\n')[0].split('\t')
            values = text.split('\n')[1].split('\t')
            return pd.Series(values, index=columns_annotation_hashes)
        
        elif (file_name == "assembly_summary.txt"):
            columns_assembly_summary = text.strip().split('\n')[1].split('\t')
            values = text.split('\n')[2].split('\t')
            return pd.Series(values, index=columns_assembly_summary)
    
        elif (file_name == "assembly_summary_historical.txt"):
            columns_assembly_summary_historical = text.strip().split('\n')[1].split('\t')
            values = text.split('\n')[2].split('\t')
            return pd.Series(values, index=columns_assembly_summary_historical)

