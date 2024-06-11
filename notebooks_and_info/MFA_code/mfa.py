import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import linregress
import pandas as pd

class MFA:

    CORNER_DICT = {'A': (0,0), 'T': (1,0), 'C': (0,1), 'G': (1,1)}
    matrix_size = 8192  # 2^13

    def __init__(self, seq):
        self.size_matrix_dict = {}
        self.seq = seq
        self.cgr_powers_matrix = None

    def get_size_matrix_dict(self):
        return self.size_matrix_dict

    @staticmethod
    def cgr_tuples(seq):
        last_point = np.array([0.5, 0.5])
        points = []

        for base in seq:
            corner = MFA.CORNER_DICT.get(base)
            if corner is None:
                continue
            last_point = (last_point + corner) / 2
           
            points.append(tuple(last_point))
        return points

    @staticmethod
    def cgr_tuples_plot(points, point_size=5):
        x, y = zip(*points)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.scatter(x, y, s=point_size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    
    @staticmethod
    def cgr(seq, m_size = None, cumulative = False):
        if m_size is None:
            print("MFA.cgr(): m_size is None")
            return None
            
        if cumulative:
            m = np.zeros((m_size, m_size), dtype=int)
        else:
            m = np.full((m_size, m_size), False)
        
        last_point = np.array([0.5, 0.5])
        
        for base in seq:
            corner = MFA.CORNER_DICT.get(base)
            if corner is None:
                continue
            last_point = (last_point + corner) / 2
            x, y = last_point * m_size

            x = int(x)
            y = int(y)
            
            # When points are getting to close to a corner
            # to the point they get an x or y value of 1,
            # they are placed in the last row or column
            if (x == m_size):
                x -= 1
            if (y == m_size):
                y -= 1
            m[ (m_size - y - 1), x] += 1
        return m
    

    # Computes a CGR matrix with a power of 2 divisions on each axis.
    # This grants the possibility of computing a new CGR matrix with
    # the half of divisions on each axis.
    def cgr_powers(self, power = 13, cumulative = True):
        m_size = np.power(2, power)
        m = MFA.cgr(self.seq, m_size, cumulative)
        self.cgr_powers_matrix = m
        return m
        
    
    # Computes a CGR matrix with the half of divisions on each axis
    # from a previous CGR matrix with a power of 2 divisions on 
    # each axis.
    def cgr_next_power(self, m = None):
        if m is None:
            if self.cgr_powers_matrix is None:
                print("mfa.cgr_next_power(): Matrix is None")
                return None
            m = self.cgr_powers_matrix
        m_size = m.shape[0]
        if m_size == 1:
            print("Matrix size is already 1")
            return None
        if m_size % 2 != 0:
            print(f"Matrix size is odd, size: {m_size}")
            return None

        new_m = np.zeros((m_size // 2, m_size // 2), dtype=int)
        # Iterate over the original matrix
        for i in range(m_size):
            for j in range(m_size):
                # Sum the values of the 4 cells that correspond to the new cell
                new_i = i // 2
                new_j = j // 2
                new_m[new_i, new_j] += m[i, j]
        self.cgr_powers_matrix = new_m
        return new_m






    @staticmethod
    def plot_cgr(m, color='black'):
        # Create a binary colormap: white for 0, "color" for any other value.
        cmap = ListedColormap(['white', color])

        # Create a normalization that maps 0 to 0 and any value >0 to 1.
        # This means all non-zero values will use the second color of the colormap.
        norm = plt.Normalize(vmin=0, vmax=1)

        # Apply a threshold to the matrix to create a binary effect:
        # 0 remains 0, anything >0 becomes 1
        binary_m = (m > 0).astype(int)

        plt.imshow(binary_m, cmap=cmap, norm=norm, extent=[0, 1, 0, 1], interpolation='nearest')
        for label, position in MFA.CORNER_DICT.items():
            plt.text(position[0], position[1], label, ha='center', va='center', color='blue')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.show()


    # Box counting method
    @staticmethod
    def calc_i(m, big_m, q):
        if big_m == 0:
            return 0  

        nonzero_mask = (m != 0)
        nonzero_vals = m[nonzero_mask]
        # for p in (nonzero_vals / big_m) ** q:
            # if p > 1:
                # print(f"p > 1; p = {p}; q = {q}; m.shape = {m.shape}; ")
        i = np.sum((nonzero_vals / big_m) ** q)
        return i


    # This method is meant to be invoked from the calc_Dq() method
    def calc_tau_q(self, epsilon_range, q_range, use_powers, 
                   plot_log_i_log_e = False):
        if epsilon_range[0] == 0:
            epsilon_range = epsilon_range[1:]
        i_mat = np.zeros(( len(epsilon_range) * len(q_range), 3), dtype=np.double)
        epsilon_used = np.zeros( (len(epsilon_range), 1))
        for idx_e, epsilon in enumerate(epsilon_range):
            m_size = round(1 / epsilon)
            if use_powers:
                if idx_e == 0:
                    m = self.cgr_powers_matrix
                else:
                    m = self.cgr_next_power()
            else:
                m = MFA.cgr(self.seq, m_size, cumulative=True) 
    
            
            big_m = np.sum(m)
            epsilon_used[idx_e] = ( 1 / m_size )

            for idx_q, q in enumerate(q_range):
                i = MFA.calc_i(m, big_m, q)
                i_mat[(idx_e * len(q_range))+ idx_q, 0] = i
                i_mat[(idx_e * len(q_range))+ idx_q, 1] = q
                i_mat[(idx_e * len(q_range))+ idx_q, 2] = epsilon_used[idx_e]
                
                if  plot_log_i_log_e:
                    plt.scatter( np.log(epsilon_used[idx_e]), np.log(i) )

        df_tau_r2 = pd.DataFrame(columns=['Q', 'Tau(Q)', 'r_squared'])
        slopes = []
        r_squared_vals = []
        for idx_q, q in enumerate(q_range):    
            # Linear regression to find slope
            i_mask = (i_mat[:, 1] == q)
            i_arr_current_q = i_mat[i_mask][:, 0]
            log_i = np.log(i_arr_current_q)
            log_epsilon = np.log(epsilon_used).flatten()

            # Filter out any NaN or infinite values that can cause errors in linregress
            valid_indices = ~(np.isnan(log_epsilon) | np.isnan(log_i) | 
                              np.isinf(log_epsilon) | np.isinf(log_i))
            slope, intercept, r_value, p_value, std_err = linregress(
                log_epsilon[valid_indices], log_i[valid_indices])
            
            slopes.append(slope)
            r_squared_vals.append(r_value**2)

            if plot_log_i_log_e:
                plt.plot( log_epsilon, log_i)
                # label for each plot
                plt.text(log_epsilon[0], log_i[0], 'q = ' + str(q))

            df_tau_r2 = pd.concat([df_tau_r2, pd.DataFrame(
                {'Q': [q], 'Tau(Q)': [slope], 'r_squared': [r_value**2]})], ignore_index=True)
        
        if plot_log_i_log_e:
            plt.xlabel('log(epsilon)')
            plt.ylabel('log(i)')
            plt.show()

        # tau_vals = np.array(slopes)
        # r_squared_vals = np.array(r_squared_vals)

        # return tau_vals, r_squared_vals, i_mat, epsilon_used
        # df_tau_r2.columns = ['Q', 'Tau(Q)', 'r_squared']
        return df_tau_r2
    


    def calc_Dq(self, epsilon_range, q_range, plot_gds = True, 
                plot_log_i_log_e = False,
                plot_cgr = False,
                use_powers = True, power = 13):
        
        point_size = 0.0001

        if use_powers:
            self.cgr_powers_matrix = self.cgr_powers(power, cumulative = True)
            epsilon_range = [ (1 / np.power(2, i)) for i in range(power, 0, -1)]
            if plot_cgr:
                # MFA.plot_cgr(self.cgr_powers_matrix, color='blue')
                MFA.cgr_tuples_plot( MFA.cgr_tuples(self.seq), point_size)
        else:
            # m_size = 50
            # m = MFA.cgr(self.seq, m_size, cumulative=True) 
            if plot_cgr:
                # MFA.plot_cgr(m, color='blue')
                MFA.cgr_tuples_plot(MFA.cgr_tuples(self.seq), point_size)


        
        # tau_vals, r_squared_vals, i_mat, epsilon_used = \
        #     self.calc_tau_q(epsilon_range, q_range, use_powers, plot_log_i_log_e)
        # df_tau_r2.columns == ['Q', 'Tau(Q)', 'r_squared']
        df_tau_r2 = self.calc_tau_q(epsilon_range, q_range, use_powers, 
                                    plot_log_i_log_e)
        
        # Define a threshold to avoid division by values too close to zero
        threshold = 1e-6
        # valid_indices = np.where(np.abs(q_range - 1) > threshold)
        # valid_q_vals = (np.abs(df_tau_r2['Tau(Q)']) - 1) > threshold
        df_valid_q_vals = df_tau_r2.loc[(np.abs(df_tau_r2['Q'] - 1)) > threshold]

        # Extract the actual indices array from the tuple
        # valid_indices = valid_indices[0]

        # D_q_vals = tau_vals[valid_indices] / (q_range[valid_indices] - 1)
        
        # D_q_vals = df_tau_r2[ valid_q_vals / (valid_q_vals - 1)]
        D_q_vals =  df_valid_q_vals['Tau(Q)'] / (df_valid_q_vals['Q'] - 1)

        df_DQ = df_valid_q_vals
        df_DQ['D(Q)'] = D_q_vals

        if plot_gds:
            # plt.scatter(q_range[valid_indices], D_q_vals)
            plt.scatter(df_valid_q_vals['Q'], D_q_vals)
            plt.xlabel('q')
            plt.ylabel('D(q)')
            # plt.plot(q_range[valid_indices], D_q_vals)
            plt.plot(df_valid_q_vals['Q'], D_q_vals)
            plt.grid()
            plt.axvline(x=0, color='black', linewidth=1)
            plt.axhline(y = np.max(D_q_vals), color='red', linewidth=1)
            # numeric label
            plt.text(0.1, np.max(D_q_vals) + 0.01, 
                     'max D(q) = ' + str(np.round(np.max(D_q_vals), 3)) )
            plt.axhline(y = np.min(D_q_vals), color='blue', linewidth=1)
            # numeric label
            plt.text(0.1, np.min(D_q_vals) - 0.03, 
                     'min D(q) = ' + str(np.round(np.min(D_q_vals), 3)) )
            plt.show()

  
        # return D_q_vals, r_squared_vals
        # df_DQ.columns == ['Q', 'Tau(Q)', 'D(Q)', 'r_squared']
        return df_DQ
    

    # GC content
    def gc_content(self, from_idx = 0, to_idx = None):
        if to_idx is None:
            to_idx = len(self.seq)
        gc_count = 0
        for base in self.seq[from_idx:to_idx]:
            if base == 'G' or base == 'C':
                gc_count += 1
        return gc_count / (to_idx - from_idx)