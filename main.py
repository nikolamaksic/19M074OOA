# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
import time

# parameters
axis_size = 200
pop_size = 30
num_of_cubes = 20

  
def draw_space(cubes):  
    # Create axis
    axes = [120, 120, 120]
      
    # Create Data
    data = np.zeros(axes, dtype=np.bool)
    for k,v in cubes.items():
        data[v[0]:v[0]+k,v[1]:v[1]+k,v[2]:v[2]+k] = True
      
    # Controll Tranperency
    alpha = 0.9
      
    # Control colour
    colors = np.empty(axes + [4], dtype=np.float32)
      
    colors[:] = [1, 0, 0, alpha]  # red
      
    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
      
    # Voxels is used to customizations of the
    # sizes, positions and colors.

    ax.voxels(data, facecolors=colors, edgecolors='grey')
    

def check_overlap(cubes,k):
    pass
    
    
    
def create_population(pop_size):
    popul = []
    for _ in range(pop_size):
        popul.append(create_individual())
    return popul

def calculate_fitness_for_individual(cubes):
    x_min,y_min,z_min = 0,0,0
    x_max,y_max,z_max = np.inf,np.inf,np.inf
    x_min_a,y_min_a,z_min_a = [],[],[]
    x_max_a,y_max_a,z_max_a = [],[],[]
    
    for k,v in cubes.iteritems():
        x_min_a.append(v[0])
        y_min_a.append(v[1])
        z_min_a.append(v[2])
        x_max_a.append(v[0]+k)
        y_max_a.append(v[1]+k)
        z_max_a.append(v[2]+k)
    x_min, y_min, z_min = min(x_min_a),min(y_min_a),min(z_min_a)
    x_max, y_max, z_max = min(x_max_a),min(y_max_a),min(z_max_a)
    V = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
    return  V

def crossover(cub1, cub2):
    
    pass

def fix_child(cub):
    # here fix overlaping
    pass


def mutate(cub):
    pass
    
    
def generate_next_population(popul):
    fitness_f = np.zeros(popul)
    for ind,p in enumerate(popul):
        fitness_f[ind] = p
    
    # norm_fit = fitness_f/np.max(fitness_f)
    norm_fit = fitness_f/sum(fitness_f)
    # here chose individ to crossover
    
    
    
    new_pop = []
    
    
    return new_pop
      
    
    
def create_individual():
    cubes = {}
    space_size = [axis_size, axis_size, axis_size]
    k = 1
    while k<=num_of_cubes:
        cubes[k] = np.random.randint(100, size=3)
        if(check_overlap(cubes,k)):
            k +=1
        
    return cubes
        

if __name__=="__main__":
    c = create_individual()
    draw_space(c)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    