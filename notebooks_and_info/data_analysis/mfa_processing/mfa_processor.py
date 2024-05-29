from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os

DIR_SEQUENCE_EXTRACTION = Path.cwd().parent / 'sequence_extraction'
print(DIR_SEQUENCE_EXTRACTION.exists())

sys.path.append(str(DIR_SEQUENCE_EXTRACTION))
from decompress_gz_files import decompress_gz_files

DIR_MFA = Path.cwd().parent.parent / 'MFA_code'
print(DIR_MFA.exists())

sys.path.append(str(DIR_MFA))
from mfa import MFA



class MFA_PROCESSOR:

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
            print(f"No .fna files found in {path_organism}")
            return None
        elif len(list_fna_files) > 1:
            print(f"Multiple .fna files found in {path_organism}")
            return None
        else:
            return list_fna_files[0]


    # Extract the sequence from the decompressed .fna file.
    # Return the nucleotide sequence and the metadata
    def extract_sequence(self, path_fna: Path):
        if path_fna == None:
            print("The path to the .fna file is None.")
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
    
    # Compute and return the GC content of a sequence, Dq values, r_squared and Delta_Dq.
    # This method ensures that the q_range includes the values [-1, 0, 1].
    def compute_gc_Dq(self, seq, epsilon_range = np.linspace(0, 0.1, 15),
                    q_range = np.linspace(-20, 20, 9), plot_gds = False, 
                    plot_log_i_log_e = False, use_powers = True, power = 13):
        q_range = np.append(q_range, [-1, 0, 1])
        q_range = np.unique(q_range)
        q_range.sort()
        instance_mfa = MFA(seq)
        # GC content
        gc_content = instance_mfa.gc_content()
        
        if use_powers:
            epsilon_range = [ (1 / np.power(2, i)) for i in range(power, 0, -1)]
        # Compute Dq
        Dq_vals, r_squared = instance_mfa.calc_Dq(epsilon_range, q_range, 
                                plot_gds, plot_log_i_log_e, use_powers, power)
        Delta_Dq = np.max(Dq_vals) - np.min(Dq_vals)
        return gc_content, Dq_vals, r_squared, Delta_Dq
        

    def compute_gc_mfa_from_list(self, directory_paths, csv_destiny_path):
        df_results = pd.DataFrame(columns=['Organism', 'path', 'seq_length', 'GC_content', 'Q', 
                'Tau(Q)', 'D(Q)_val', 'r_squared_vals', 'Delta_D(Q)']).astype({
            'Organism': 'str',
            'path': 'str',
            'seq_length': 'Int64',
            'GC_content': 'float64',
            'Q': 'Int64',
            'Tau(Q)': 'float64',
            'D(Q)_val': 'float64',
            'r_squared_vals': 'float64',
            'Delta_D(Q)': 'float64'
        })
        for dir in directory_paths:
            # dir_files = dir.glob('*')
            organism_name = dir.name
            print(f"processing: {organism_name}")
            fna_path = self.get_path_fna(dir)
            # for file_path in dir_files:
            seq, seq_metadata = self.extract_sequence(fna_path)
            gc_content, Dq_vals, r_squared_vals, Delta_Dq = self.compute_gc_Dq(seq)
            
            r_squared_vals = np.delete( r_squared_vals, len(r_squared_vals)//2 )
            print(f"type(r_squared_vals): {type(r_squared_vals)}")
            print(f"content: {gc_content}")
            print(f"len(Dq_vals): {len(Dq_vals)}")
            print(f"len(r_squared_vals): {len(r_squared_vals)}")
            print(f"Delta_Dq: {Delta_Dq}")

            data = {
                'Organism': [organism_name] * len(Dq_vals),
                'path': [dir] * len(Dq_vals),
                'seq_length': [len(seq)] * len(Dq_vals),
                'GC_content': [gc_content] * len(Dq_vals),
                'Q': [pd.NA] * len(Dq_vals),
                'Tau(Q)': [pd.NA] * len(Dq_vals),
                'D(Q)_val': Dq_vals,
                'r_squared_vals': r_squared_vals,
                'Delta_D(Q)': [Delta_Dq] * len(Dq_vals)
            }
            df_organism = pd.DataFrame(data)
            # df_results = pd.concat([df_results, df_organism], ignore_index=True)
            if os.path.exists(csv_destiny_path):
                df_organism.to_csv(csv_destiny_path, mode='a', header=False, index=False)
            else:
                df_organism.to_csv(csv_destiny_path, mode='w', header=True, index=False)



    def delete_genomes(self, directory_paths):
        for dir in directory_paths:
            files = dir.glob('*')
            for file in files:
                if file.name.endswith('.fna') or file.name.endswith('.fna.gz'):
                    file.unlink()

