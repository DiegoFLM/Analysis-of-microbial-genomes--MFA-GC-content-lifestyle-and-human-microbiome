from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os

DIR_SEQUENCE_EXTRACTION = Path.cwd().parent / 'sequence_extraction'
print(DIR_SEQUENCE_EXTRACTION.exists())

sys.path.append(str(DIR_SEQUENCE_EXTRACTION))
from decompress_gz_files import decompress_gz_files, decompress_tgz_files

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
        decompress_tgz_files(self.ROOT_PATH, include_dirs)


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
        
        # if there is no file, return None
        if not path_fna.exists():
            print(f"The file {path_fna} does not exist.")
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
                    q_range = np.linspace(-20, 20, 41), plot_gds = False, 
                    plot_log_i_log_e = False, plot_cgr = False, 
                    use_powers = True, power = 13):
        q_range = np.append(q_range, [-1, 0, 0.99999])
        q_range = np.unique(q_range)
        q_range.sort()
        instance_mfa = MFA(seq)
        # GC content
        gc_content = instance_mfa.gc_content()
        
        if use_powers:
            epsilon_range = [ (1 / np.power(2, i)) for i in range(power, 0, -1)]

        df_DQ = instance_mfa.calc_Dq(epsilon_range, 
                                    q_range, 
                                    plot_gds, 
                                    plot_log_i_log_e = plot_log_i_log_e, 
                                    plot_cgr = plot_cgr, 
                                    use_powers = use_powers, 
                                    power = power)
        df_DQ['Delta_Dq'] = df_DQ['D(Q)'].max() - df_DQ['D(Q)'].min()
        df_DQ['GC_content'] = gc_content

        # df_DQ.columns = ['Q', 'Tau(Q)', 'D(Q)', 'r_squared', 'Delta_Dq', 'GC_content']
        return df_DQ

    def compute_gc_mfa_from_list(self, directory_paths, csv_destiny_path, 
                                 epsilon_range = np.linspace(0, 0.1, 15),
                                 q_range = np.linspace(-20, 20, 41),
                                 plot_gds = False, 
                                 plot_log_i_log_e = False, 
                                 plot_cgr = False,
                                 use_powers = True, 
                                 power = 13):
        

        for current_path in directory_paths:
            if current_path.is_dir():
                organism_name = current_path.name
                fna_path = self.get_path_fna(current_path)
                seq, seq_metadata = self.extract_sequence(fna_path)
            elif current_path.is_file() and current_path.name.endswith('.fna'):
                organism_name = current_path.name[:-4]
                seq, seq_metadata = self.extract_sequence(current_path)
            else:
                print(f"Invalid path: {current_path}")
                continue
                
            print(f"processing: {organism_name}")

            df_results = pd.DataFrame(columns=['Organism', 'path', 'seq_length', 'GC_content', 'Q', 
                    'Tau(Q)', 'D(Q)', 'r_squared', 'Delta_Dq']).astype({
                'Organism': 'str',
                'path': 'str',
                'seq_length': 'Int64',
                'GC_content': 'float64',
                'Q': 'Int64',
                'Tau(Q)': 'float64',
                'D(Q)': 'float64',
                'r_squared': 'float64',
                'Delta_Dq': 'float64'
            })

            # df_DQ.columns == ['Q', 'Tau(Q)', 'D(Q)', 'r_squared', 'Delta_Dq', 'GC_content']
            df_DQ = self.compute_gc_Dq(seq, 
                                       epsilon_range = epsilon_range,
                                       q_range = q_range,
                                       plot_gds = plot_gds, 
                                       plot_log_i_log_e = plot_log_i_log_e,
                                       plot_cgr = plot_cgr,
                                       use_powers = use_powers, 
                                       power = power)
            df_DQ['Organism'] = organism_name
            if current_path.is_dir():
                df_DQ['path'] = current_path
            elif current_path.is_file():
                df_DQ['path'] = current_path.parent
            df_DQ['seq_length'] = len(seq)
            df_DQ = df_DQ[['Organism', 'path', 'seq_length', 'GC_content', 'Q', 'Tau(Q)', 
                           'D(Q)', 'r_squared', 'Delta_Dq']]
            

            df_results = pd.concat([df_results, df_DQ], ignore_index=True)




            # # START SEGMENTS

            # for i in range(3):
            #     segment_length = int(len(seq)//3)
            #     segment = seq[ i * segment_length : (i+1) * segment_length]
            #     df_segments = self.compute_gc_Dq(segment, 
            #                             epsilon_range = epsilon_range,
            #                             q_range = q_range,
            #                             plot_gds = plot_gds, 
            #                             plot_log_i_log_e = plot_log_i_log_e,
            #                             plot_cgr = plot_cgr,
            #                             use_powers = use_powers, 
            #                             power = power)
            #     df_segments['Organism'] = organism_name + f'_segment_{i+1}'
            #     print(f"processing: {organism_name + f'_segment_{i+1}'}")
            #     if current_path.is_dir():
            #         df_segments['path'] = current_path
            #     elif current_path.is_file():
            #         df_segments['path'] = current_path.parent
            #     df_segments['seq_length'] = len(segment)
            #     df_segments = df_segments[['Organism', 'path', 'seq_length', 'GC_content', 'Q', 'Tau(Q)', 
            #                 'D(Q)', 'r_squared', 'Delta_Dq']]
            #     df_results = pd.concat([df_results, df_segments], ignore_index=True)
            # # END SEGMENTS




            if os.path.exists(csv_destiny_path):
                df_results.to_csv(csv_destiny_path, mode='a', header=False, index=False, sep=';', decimal=',')
            else:
                df_results.to_csv(csv_destiny_path, mode='w', header=True, index=False, sep=';', decimal=',')




    def delete_genomes(self, directory_paths):
        for dir in directory_paths:
            files = dir.glob('*')
            for file in files:
                if file.name.endswith('.fna') or file.name.endswith('.fna.gz'):
                    file.unlink()

