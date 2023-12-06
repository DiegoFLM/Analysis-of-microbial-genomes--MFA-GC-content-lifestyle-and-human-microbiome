import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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


    def plot_tau_q(self, epsilon_range, q_range):
        if epsilon_range[0] == 0:
            epsilon_range = epsilon_range[1:]
        print(epsilon_range)
        i_mat = np.zeros(( len(epsilon_range) * len(q_range), 3), dtype=np.double)
        epsilon_used = np.zeros(( len(epsilon_range), 1))
        for idx_e, epsilon in enumerate(epsilon_range):
            m_size = round(1 / epsilon)
            m = MFA.cgr(self.seq, m_size)   
            epsilon_used[idx_e] = ( 1 / m_size )

            for idx_q, q in enumerate(q_range):
                i = MFA.calc_i(m, q)
                i_mat[(idx_e * len(q_range))+ idx_q, 0] = i
                i_mat[(idx_e * len(q_range))+ idx_q, 1] = q
                i_mat[(idx_e * len(q_range))+ idx_q, 2] = epsilon_used[idx_e]
                
                plt.scatter( np.log(epsilon_used[idx_e]), np.log(i) )
        print(i_mat)
        print(epsilon_used)
        for idx_q, q in enumerate(q_range):    

            # Why?
            # if q == 0:
            #     continue
            i_mask = (i_mat[:,1] == q)
            i_arr_current_q = i_mat[i_mask][:,0]
            
            plt.plot( [np.log(e) for e in epsilon_used], 
                     np.log(i_arr_current_q))
            # labels for axis
            plt.xlabel('log(epsilon)')
            plt.ylabel('log(i)')
        plt.show()
        return i_mat, epsilon_used
    

        