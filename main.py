# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
import math as ma
import time
import copy

# parameters
axis_size = 100
# popul_size = 300
# num_of_alg_iter = 1000
popul_size = 200
num_of_alg_iter = 10000
prob_to_mutate = 0.1
cub_list = [x for x in range(1,21)]
im_folder = r'.\img\\'

def find_space_bounds(cubes):
    x_min,y_min,z_min = 0,0,0 
    x_max,y_max,z_max = np.inf,np.inf,np.inf
    x_min_a,y_min_a,z_min_a = [],[],[]
    x_max_a,y_max_a,z_max_a = [],[],[]
    
    for k,v in cubes.items():
        x_min_a.append(v[0])
        y_min_a.append(v[1])
        z_min_a.append(v[2])
        x_max_a.append(v[0]+k)
        y_max_a.append(v[1]+k)
        z_max_a.append(v[2]+k)
    x_min, y_min, z_min = min(x_min_a),min(y_min_a),min(z_min_a)
    x_max, y_max, z_max = max(x_max_a),max(y_max_a),max(z_max_a)

    
    return x_min, y_min, z_min, x_max, y_max, z_max

def draw_and_save_space(cubes,it):  
    # Create axis
    x_min, y_min, z_min, x_max, y_max, z_max = find_space_bounds(cubes)
    
    x_size = x_max-x_min
    y_size = y_max-y_min
    z_size = z_max-z_min
    ax_s = max([x_size, y_size,z_size])
    # axes = [x_size, y_size, z_size]
    axes = [ax_s, ax_s, ax_s]
      
    # Create Data
    data = np.zeros(axes, dtype=np.bool)
    for k,v in cubes.items():
        # draw all cubes shifted by the min position; this is usefull because 
        # the bigger space is the more time function need to draw whole space
        data[v[0]-x_min:v[0]+k-x_min,v[1]-y_min:v[1]+k-y_min,v[2]-z_min:v[2]+\
             k-z_min] = True 
      
    # Controll Tranperency
    alpha = 0.9
      
    # Control colour
    colors = np.empty(axes + [4], dtype=np.float32)
      
    colors[:] = [1, 0, 0, alpha]  # red
      
    # Plot figure
    
    fig = plt.figure()
    # plt.ion()
    
    # plt.ioff()
    ax = fig.add_subplot(111, projection='3d')
    # Voxels is used to customizations of the
    # sizes, positions and colors.

    ax.voxels(data, facecolors=colors, edgecolors='grey')
    plt.show()
    
    # plt.ioff()
    plt.savefig(im_folder+'\\'+str(it)+'.PNG',bbox_inches='tight')  

def check_two_cubes_overlap(cubeA,cubeA_size,cubeB,cubeB_size):
    ret = (cubeA[0]+cubeA_size > cubeB[0] and cubeA[0] < cubeB[0]+cubeB_size and\
               cubeA[1]+cubeA_size > cubeB[1] and cubeA[1] < cubeB[1]+cubeB_size and\
               cubeA[2]+cubeA_size > cubeB[2] and cubeA[2] < cubeB[2]+cubeB_size)
    return ret

def current_cube_position_overlap(cubes,k):
    """
    Check if the given cubes is overlapping other cubes in array.

    Parameters
    ----------
    cubes : {}
        all cubes generated for current individual.
    k : int
        current cube.

    Returns
    -------
    bool
        True if the cube overlaps other ones.

    """
    if(k==cub_list[0]):
        return False
    comp_c = cubes[k]
    for cub_s,c in cubes.items():
        if(cub_s!=k):
            # if(cubes[k][0]+k > c[0] and cubes[k][0] < c[0]+cub_s and\
            #    cubes[k][1]+k > c[1] and cubes[k][1] < c[1]+cub_s and\
            #    cubes[k][2]+k > c[2] and cubes[k][2] < c[2]+cub_s):
            if(check_two_cubes_overlap(cubes[k],k,c,cub_s)):
                return True

    return False
    
def check_two_array_cubes_overlaping(c1, c2):
    for k1, v1 in c1.items():
        for k2, v2 in c2.items():
            if(check_two_cubes_overlap(v1,k1,v2,k2)):
                return True
    return False

def create_population(pop_size):
    """
    Create initial population with given size

    Parameters
    ----------
    pop_size : int
        num of individuals in pop size.

    Returns
    -------
    popul : []
        list of individuals.

    """
    popul = []
    for _ in range(pop_size):
        popul.append(create_individual())
    return popul

def calculate_fitness_for_individual(cubes):
    """
    Calculate fitness function for individual, the fitness is calculated
    as 1/V where V is volume of the required space 

    Parameters
    ----------
    cubes : {}
        all cubes for the give.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x_min,y_min,z_min = 0,0,0
    x_max,y_max,z_max = np.inf,np.inf,np.inf
    x_min_a,y_min_a,z_min_a = [],[],[]
    x_max_a,y_max_a,z_max_a = [],[],[]
    
    for k,v in cubes.items():
        x_min_a.append(v[0])
        y_min_a.append(v[1])
        z_min_a.append(v[2])
        x_max_a.append(v[0]+k)
        y_max_a.append(v[1]+k)
        z_max_a.append(v[2]+k)
    x_min, y_min, z_min = min(x_min_a),min(y_min_a),min(z_min_a)
    x_max, y_max, z_max = max(x_max_a),max(y_max_a),max(z_max_a)
    # V = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
    V = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)-80000
    return  1/V # biger fitness = better individua

def find_best_individual(p):
    """
    Find the best individuak in the given population and its fitnes

    Parameters
    ----------
    p : []
        population: list of individuals.

    Returns
    -------
    max_fit : float
        max fitness function.
    indiv : {}
        Individual.

    """
    max_fit = 0
    ind = -1
    for i, k in enumerate(p):
        new_max = calculate_fitness_for_individual(k)
        if(new_max>max_fit):
            max_fit = new_max
            indiv = k
            ind = i
    
    assert max_fit!=0
    # V = int(round(1/max_fit)) # Volume
    V = int(round(1/max_fit))+80000 # Volume
    return indiv, ind, V

def append_second_parent_features_to_child(current_ch, sec_par_cub):
    new_ch = current_ch
    x_min_ch, y_min_ch, z_min_ch, x_max_ch, y_max_ch, z_max_ch = find_space_bounds(current_ch)
    x_min_par, y_min_par, z_min_par, x_max_par, y_max_par, z_max_par = find_space_bounds(sec_par_cub)
    for k in sec_par_cub:
        sec_par_cub[k][0] = sec_par_cub[k][0] + x_min_ch - x_min_par
        sec_par_cub[k][1] = sec_par_cub[k][1] + y_min_ch - y_min_par
        sec_par_cub[k][2] = sec_par_cub[k][2] + z_min_ch - z_min_par
    pom_sec_par = copy.deepcopy(sec_par_cub)
    if(not check_two_array_cubes_overlaping(sec_par_cub, current_ch)):
        for k, v in pom_sec_par.items():
            new_ch[k] = v
        return new_ch
    for _ in range(x_min_ch, x_max_ch + 1):
        for k in pom_sec_par:
            pom_sec_par[k][0] += 1
            pom_sec_par[k][1] = sec_par_cub[k][1]
        
        for _ in range(y_min_ch, y_max_ch + 1):
            
            for k in pom_sec_par:
                pom_sec_par[k][1] += 1
                pom_sec_par[k][2] = sec_par_cub[k][2]
            
            for _ in range(y_min_ch, y_max_ch + 1):
                for k in pom_sec_par:
                    pom_sec_par[k][2] += 1
                if(not check_two_array_cubes_overlaping(pom_sec_par, current_ch)):
                    for k, v in pom_sec_par.items():
                        new_ch[k] = v
                    return new_ch
                      
    assert False     
                    
                    
    

def crossover(par1, par2):
    # d = np.random.randint(1, len(par1)-1)
    cross_pos = np.random.randint(1, len(par1)-1)
    
    # cross_pos = int(round(len(cub_list)/d))
    ch1 = {}
    ch2 = {}
    ind = 0
    p1 = {}
    p2 = {}
    for k in par1:
        if(ind<=cross_pos):
            ch1[k] = par1[k]
            ch2[k] = par2[k]
        else:
            p1[k]  = par1[k]
            p2[k]  = par2[k]
        ind+=1
    
    ch1 = append_second_parent_features_to_child(ch1,p2)
    ch2 = append_second_parent_features_to_child(ch2,p1)

    return ch1, ch2




def mutate(cub):
    cub_to_mut = np.random.choice(cub_list)
    finished = False
    while(not finished):
        cub[cub_to_mut] = np.random.randint(axis_size//2, size=3) # xmin, ymin, zmin
        # check if we found place where new cube is not overlaping other ones
        if(not current_cube_position_overlap(cub,cub_to_mut)): 
            finished = True
    return cub
    
def find_parent_index(popul,fitness_arr):
    p = np.random.uniform(0, 1)
    k = 0
    while(p>=fitness_arr[k]):
        k+=1
        if(k==(len(fitness_arr)-1)):
            break
    # print('Probability:{},{}'.format(p,fitness_arr[k]))
    return k
        
    
    
    
def generate_next_population(popul):
    fitness_f = np.zeros(len(popul))
    new_pop = []
    for ind,p in enumerate(popul):
        fitness_f[ind] = calculate_fitness_for_individual(p)
        
        
    
    norm_fit = fitness_f/sum(fitness_f)
    fit_array = np.zeros(len(popul))
    fit_sum = 0
    for ind in range(len(fit_array)):
        fit_sum += norm_fit[ind]
        fit_array[ind] = fit_sum
    # dont give multiple cross with same parents
    for k in range(int(round(len(popul)/2))):
        par1_ind = find_parent_index(popul,fit_array)
        par2_ind = find_parent_index(popul,fit_array)
        # par1_ind = k
        # par2_ind = int(round(len(popul)/2))+k-1
        
        if(par1_ind==par2_ind):
            while(par1_ind==par2_ind):
                par2_ind = find_parent_index(popul,fit_array)
        
        par1 = popul[par1_ind]
        par2 = popul[par2_ind]
        # new_pop.append(par1)
        # new_pop.append(par2)
        new_ch1,new_ch2 = crossover(par1,par2)
        new_pop.append(new_ch1)
        new_pop.append(new_ch2)
    
    
    if(prob_to_mutate>np.random.uniform(0,1)):
        child_to_mut = np.random.randint(0, len(popul))
        popul[child_to_mut] = mutate(popul[child_to_mut])
        
    return new_pop
      
# def generate_next_population(popul):
#     fitness_f = np.zeros(len(popul))
#     new_pop = []
#     for ind,p in enumerate(popul):
#         fitness_f[ind] = calculate_fitness_for_individual(p)
        
        
#     popul = [x for _, x in sorted(zip(popul,fitness_f))]
#     k = popul_size/5
#     new_pop = popul[]
#     # dont give multiple cross with same parents
#     for k in range(int(round(len(popul)/2))):
#         par1_ind = find_parent_index(popul,fit_array)
#         par2_ind = find_parent_index(popul,fit_array)
#         # par1_ind = k
#         # par2_ind = int(round(len(popul)/2))+k-1
        
#         if(par1_ind==par2_ind):
#             while(par1_ind==par2_ind):
#                 par2_ind = find_parent_index(popul,fit_array)
        
#         par1 = popul[par1_ind]
#         par2 = popul[par2_ind]
#         # new_pop.append(par1)
#         # new_pop.append(par2)
#         new_ch1,new_ch2 = crossover(par1,par2)
#         new_pop.append(new_ch1)
#         new_pop.append(new_ch2)
    
    
#     if(prob_to_mutate>np.random.uniform(0,1)):
#         child_to_mut = np.random.randint(0, len(popul))
#         popul[child_to_mut] = mutate(popul[child_to_mut])
        
#     return new_pop   

    
def create_individual():
    # create one individual
    num_of_cubes = len(cub_list)
    cubes = {}
    k = 0
    while k<num_of_cubes:
        cubes[cub_list[k]] = np.random.randint(axis_size//2, size=3) # xmin, ymin, zmin
        # check if we found place where new cube is not overlaping other ones
        if(not current_cube_position_overlap(cubes,cub_list[k])): 
            k += 1
        
    return cubes
  
def main():
    best_invidi_per_iter = []
    best_vol_per_iter = []
    p = create_population(popul_size)
    c,ind,V = find_best_individual(p)
    print('\n')
    for k in range(num_of_alg_iter):
        p = generate_next_population(p)
        c,ind,V = find_best_individual(p)
        print('Iteration: {0: <4}/{1: <4}, the lowest volume: {2}'.format(k,num_of_alg_iter,V))
        best_invidi_per_iter.append(c)
        best_vol_per_iter.append(V)
        
        # draw_and_save_space(c,k)
    # for ind,v in enumerate(best_vol_per_iter):
    #     print("Iter: {0: <4}- volume: {1}".format(str(ind),v))
    
    # save all results into a file
    file = open("sample.txt", "w")
    for bes_ind in best_invidi_per_iter:
        file.write("%s = %s\n" %("a_dictionary", bes_ind))
    
    file.close()
    return best_vol_per_iter, best_invidi_per_iter

if __name__=="__main__":
    bv, bi = main()
    bi_sort = [x for _, x in sorted(zip(bv,bi))]
    bi_max = bi_sort[0]
    draw_and_save_space(bi_max,1001)
    
    fig = plt.figure()
    plt.plot(bv)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    