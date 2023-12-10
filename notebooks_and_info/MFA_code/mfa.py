import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import linregress

class MFA:

    CORNER_DICT = {'A': (0,0), 'T': (1,0), 'C': (0,1), 'G': (1,1)}
    matrix_size = 10000

    def __init__(self, seq):
        self.size_matrix_dict = {}
        self.seq = seq

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
            m_size = MFA.matrix_size
            
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
            m[ (m_size - y - 1), x] += 1
        return m
    

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
    def calc_i(m, q):
        big_m = np.sum(m)
        if big_m == 0:
            return 0  

        epsilon = 1 / m.shape[0]
        nonzero_mask = (m != 0)
        nonzero_vals = m[nonzero_mask]
        i = np.sum((nonzero_vals / big_m) ** q)
        return i


    def calc_tau_q(self, epsilon_range, q_range, plot_log_i_log_e = False):
        if epsilon_range[0] == 0:
            epsilon_range = epsilon_range[1:]
        i_mat = np.zeros(( len(epsilon_range) * len(q_range), 3), dtype=np.double)
        epsilon_used = np.zeros(( len(epsilon_range), 1))
        for idx_e, epsilon in enumerate(epsilon_range):
            m_size = round(1 / epsilon)
            m = MFA.cgr(self.seq, m_size, cumulative=True)   
            epsilon_used[idx_e] = ( 1 / m_size )

            for idx_q, q in enumerate(q_range):
                i = MFA.calc_i(m, q)
                i_mat[(idx_e * len(q_range))+ idx_q, 0] = i
                i_mat[(idx_e * len(q_range))+ idx_q, 1] = q
                i_mat[(idx_e * len(q_range))+ idx_q, 2] = epsilon_used[idx_e]
                
                if  plot_log_i_log_e:
                    plt.scatter( np.log(epsilon_used[idx_e]), np.log(i) )

        slopes = []
        r_squared_vals = []
        for idx_q, q in enumerate(q_range):    

            # Linear regression to find slope
            i_mask = (i_mat[:, 1] == q)
            i_arr_current_q = i_mat[i_mask][:, 0]
            log_epsilon = np.log(epsilon_used).flatten()
            log_i = np.log(i_arr_current_q)

            # Filter out any NaN or infinite values that can cause errors in linregress
            valid_indices = ~(np.isnan(log_epsilon) | np.isnan(log_i) | 
                              np.isinf(log_epsilon) | np.isinf(log_i))
            slope, intercept, r_value, p_value, std_err = linregress(
                log_epsilon[valid_indices], log_i[valid_indices])
            

            slopes.append(slope)
            r_squared_vals.append(r_value**2)

            if plot_log_i_log_e:
                plt.plot( log_epsilon, 
                        log_i)
                # label for each plot
                plt.text(log_epsilon[0], log_i[0], 'q = ' + str(q))
        
        if plot_log_i_log_e:
            plt.xlabel('log(epsilon)')
            plt.ylabel('log(i)')
            plt.show()

        tau_vals = np.array(slopes)
        r_squared_vals = np.array(r_squared_vals)

        return tau_vals, r_squared_vals, i_mat, epsilon_used


    def calc_Dq(self, epsilon_range, q_range, plot_gds = True, plot_log_i_log_e = False):
        tau_vals, r_squared_vals, i_mat, epsilon_used = \
            self.calc_tau_q(epsilon_range, q_range, plot_log_i_log_e)
        
        # Define a threshold to avoid division by values too close to zero
        threshold = 1e-6
        valid_indices = np.where(np.abs(q_range - 1) > threshold)

        # Extract the actual indices array from the tuple
        valid_indices = valid_indices[0]

        D_q_vals = tau_vals[valid_indices] / (q_range[valid_indices] - 1)

        if plot_gds:
            plt.scatter(q_range[valid_indices], D_q_vals)
            plt.xlabel('q')
            plt.ylabel('D(q)')
            plt.plot(q_range[valid_indices], D_q_vals)
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

        return D_q_vals, r_squared_vals