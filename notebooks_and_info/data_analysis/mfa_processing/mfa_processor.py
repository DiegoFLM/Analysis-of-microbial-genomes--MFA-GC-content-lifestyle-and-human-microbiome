from pathlib import Path
import numpy as np
import sys

DIR_SEQUENCE_EXTRACTION = Path.cwd().parent / 'sequence_extraction'
print(DIR_SEQUENCE_EXTRACTION.exists())

sys.path.append(str(DIR_SEQUENCE_EXTRACTION))
from decompress_gz_files import decompress_gz_files

DIR_MFA = Path.cwd().parent.parent / 'MFA_code'
print(DIR_MFA.exists())

sys.path.append(str(DIR_MFA))
from mfa import MFA



class MFA_PROCESSOR:
    ROOT_PATH = None


    # The constructor receives a Path for the directory containing the 
    # subdirectories with organisms data
    def __init__(self, root_path):
        self.ROOT_PATH = Path(root_path)


    def set_root_path(self, root_path):
        self.ROOT_PATH = Path(root_path)

    
    def decompress(self, include_dirs: list):
        decompress_gz_files(self.ROOT_PATH, include_dirs)


    def get_path_fna(self, path_organism: Path):
        if not path_organism.is_dir():
            print(f"The path {path_organism} is not a directory.")
            return None
        list_fna_files = list(path_organism.glob('*.fna'))
        if len(list_fna_files) == 0:
            print(f"No .gz files found in {path_organism}")
            return None
        elif len(list_fna_files) > 1:
            print(f"Multiple .gz files found in {path_organism}")
            return None
        else:
            return list_fna_files[0]

            
    # Extract the sequence from the decompressed .fna file 
    def extract_sequence(self, path_fna: Path):
        if path_fna == None:
            return None
        #read the file and clear it
        with open(path_fna, 'r') as f:
            fna_content = f.read()

        file_lines = fna_content.split('\n')
        sequence = ""
        metadata = ""

        for line in file_lines:
            if line.startswith('>'):  
                metadata += line
            elif ( len(line) == 0 ):
                continue
            else:
                sequence += line
        return sequence, metadata
    

    def compute_gc_mfa_from_sequence(self, seq):
        instance_mfa = MFA(seq)

        
    
    def compute_gc_mfa_from_list(self, directory_names, csv_destiny_path):
        for dir in directory_names:
            dir_files = dir.glob('*')
            for file in dir_files:
                if file.name.endswith('.fna'):
                    seq = self.extract_sequence(file)
                    self.compute_gc_mfa_from_sequence(seq)



    def delete_genomes(self, directories_list):
        for dir in directories_list:
            files = dir.glob('*')
            for file in files:
                if file.name.endswith('.fna') or file.name.endswith('.fna.gz'):
                    file.unlink()

