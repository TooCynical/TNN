import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as patches


import Util


# Grid based SOM.
class SOM:
    class Center:
        def __init__(self, N, pos=None):
            if pos != None:
                self.pos = pos
            else:
                sampler = Util.uniform_sampler(0, 1)
                self.pos = np.array([sampler() for dummy in range(N)])
            
            self.N = N


        def __call__(self, X):
            return Util.quadratic_error(self.pos, X)
            
       
        def update(self, X, delta):
            self.pos += delta * (X - self.pos)
    

        def __repr__(self):
            return str(self.N) + '-center @ ' + str(self.pos)


    def __init__(self, N=1, g=1, K=[1], F=[1], eta_0=0.3, eta_final=0., t_final=1000, size=0.99):
        if g != len(K) or g != len(F):
            raise ValueError("Grid dimension and length of K, F should match.")
        
        self.N = N
        self.g = g
        self.K = K
        self.F = F
        
        self.t = 0
        self.eta = Util.exp_decay(eta_0, eta_final, t_final)
        self.h = Util.exp_topo(size)
        
        self.__init_centers()


    # Update center position for given input X according to the SOM learning rule.
    def learn(self, X):
        # Find winning center.
        win_index = self.winner(X)[0]
        
        # Adjust all center positions.
        for index in self.grid_indices:
            self.centers[index].update(X, self.__delta(win_index, index))
        
        # Update t.
        self.t += 1
        
        
    # Returns the index of the winning center for given input, as well as the 
    # distance from this center to the input.
    def winner(self, X):
        min_distance = -1
        best_center_index = None
        for index in self.grid_indices:
            new_distance = self.centers[index](X)
            if new_distance < min_distance or min_distance == -1:
                min_distance = new_distance
                best_center_index = index
        return best_center_index, min_distance
        
        
    # Returns the locations of each center.
    def center_locs(self):
        return np.array([self.centers[index].pos for index in self.grid_indices]).T

        
    # Compute the delta for two points in the grid (delta = h(d(index1, index2))) * eta(t).
    def __delta(self, index1, index2):
        return self.h(self.__grid_distance(index1, index2)) * self.eta(self.t)

    
    # Returns the distance between two grid points given by their indices,
    # taking into account the edge weights.
    def __grid_distance(self, index1, index2):
        total_distance = 0
        for i in range(self.g):
            total_distance += abs(index1[i] - index2[i]) * self.F[i]
        return total_distance


    def __init_centers(self):
        l = [range(k) for k in self.K]
        self.grid_indices = []
        for index in itertools.product(*l):
            self.grid_indices.append(index)

        self.centers = {}
        for index in self.grid_indices:
            self.centers[index] = SOM.Center(self.N)
            
            
    def __repr__(self):
        out = "SOM ( "
        for k in self.K:
            out += str(k) + " "
        out += ")\n"
        out += "centers: \n"
        for index in self.grid_indices:
            out += str(self.centers[index]) + "\t" + str(index) + "\n"
        return out


# An MSOM is basically just a container for multiple SOMS, which decides
# which SOM should learn a certain data point when given one.
class MSOM:
   def __init__(self, N, soms):
       self.N = N
       self.soms = []
       for som in soms:
           self.add_som(som)
          
      
   def learn(self, X):
       self.__closest_som(X).learn(X)


   def add_som(self, som):
       if som.N != self.N:
           raise ValueError("SOM dimension mismatch!")
       else:
           self.soms.append(som)
           

   def __closest_som(self, X):
       min_distance = -1
       closest_som = None
       for som in self.soms:
           new_distance = som.winner(X)[1]
           if min_distance == -1 or new_distance < min_distance:
               min_distance = new_distance
               closest_som = som
       return closest_som
   
    
   def plot(self, ax, colors):
       i = 0
       for som in self.soms:
           x, y = som.center_locs()
           ax.scatter(x, y, c=colors[i])
           i += 1
       
       
   def __repr__(self):
       out = "MSOM: \n"
       for som in self.soms:
           out += str(som)
       return out
       
       
def demo(plot=False):
    som1 = SOM(2, 2, [4,4], [1,1])    
    som2 = SOM(2, 2, [4,4], [1,1])    
    som3 = SOM(2, 2, [4,4], [1,1])
    som4 = SOM(2, 2, [4,4], [1,1])
    
    msom = MSOM(2, [som1, som2, som3, som4])
    
    square1 = Util.uniform_sampler2D(0., 0.2, 0., 0.2)
    square2 = Util.uniform_sampler2D(0.5, 0.8, 0.7, 1.0)
    square3 = Util.uniform_sampler2D(0.4, 1., 0., 0.6)
    square4 = Util.uniform_sampler2D(0., 0.2, 0.8, 1.0)
    
    for i in xrange(1000):
        msom.learn(square1())
        msom.learn(square2())
        msom.learn(square3())
        msom.learn(square4())

        
    print msom

    if plot:
        patch1 = patches.Rectangle((0.0, 0.0),0.2,0.2,linewidth=1,edgecolor='r',facecolor='none')
        patch2 = patches.Rectangle((0.5, 0.7),0.3,0.3,linewidth=1,edgecolor='r',facecolor='none')
        patch3 = patches.Rectangle((0.4, 0.0),0.6, 0.6,linewidth=1,edgecolor='r',facecolor='none')
        patch4 = patches.Rectangle((0.0, 0.8),0.2,0.2,linewidth=1,edgecolor='r',facecolor='none')
        # Create figure and axes
        fig, ax = plt.subplots(1)
        
        msom.plot(ax, colors=['red', 'green', 'blue', 'yellow'])
        ax.add_patch(patch1)    
        ax.add_patch(patch2)
        ax.add_patch(patch3)
        ax.add_patch(patch4)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('equal')
    
        plt.show()
    


if __name__ =="__main__":
    demo(0)


