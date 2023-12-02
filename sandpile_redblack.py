import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import csv
from numpy.core.defchararray import count
from datetime import datetime
from numpy.lib.function_base import vectorize
import random

import time


complete_history = []
complete_history_index = []
fallout_edges= []
iteration_piles = []

stimulation=0
now = datetime.now()
folder_name = now.strftime("%b-%d-%Y.%H.%M.%S")
os.mkdir(folder_name)
fig = plt.figure(figsize=(10,5),dpi=1600)
ax = fig.add_subplot(111)
equation_list = []
equation_plot = []
scatter_plot=[]
scatter_list=[]
current_model = ""
def initialize_pile(x,y):
    '''
    
    \Parameters\: Define size by given x and y 
    x: columns of matrix 
    y: height of matrix 

    output: empty matrix with shape (y,x,4) with -1 filled inside
    '''
    matrix = np.empty((y,x),dtype=object)
    matrix.fill(np.array([],dtype=int))
    return matrix

def drop_sand(pile_container,id):
    '''
    Paramters: matrix the sand will be drop in and id for current drop
    
    pile_container: expected matrix to be shape of (y,x,4)
    id: int represents sand id

    output: updated the sand pile with id updated.
    '''
    x = pile_container.shape[1]
    y = pile_container.shape[0]
    location = (np.random.randint(0,y,(1,)).flatten()[0],np.random.randint(0,x,(1,)).flatten()[0])
    current_container = pile_container[location[0]][location[1]]
    
    coin_flip = np.random.randint(0,2,size=(1,)).flatten()[0]

    if coin_flip == 0:
        id = 0
    else:
        id = 1

    current_container = np.append(current_container,id)
    pile_container[location[0]][location[1]] = current_container

    return pile_container
def drop_sand_switch(pile_container,id,red_prob):
    '''
    Paramters: matrix the sand will be drop in and id for current drop
    
    pile_container: expected matrix to be shape of (y,x,4)
    id: int represents sand id

    output: updated the sand pile with id updated.
    '''
    x = pile_container.shape[1]
    y = pile_container.shape[0]
    location = (np.random.randint(0,y,(1,)).flatten()[0],np.random.randint(0,x,(1,)).flatten()[0])
    current_container = pile_container[location[0]][location[1]]
    
    coin_flip = random.uniform(0,1)
    if coin_flip<=red_prob:
        id = 0
    else:
        id = 1

    current_container = np.append(current_container,id)
    pile_container[location[0]][location[1]] = current_container

    return pile_container
def find_pos(matrix):
    positions = []
    vector_x = np.vectorize(lambda arr:arr.shape[0])
    mask = vector_x(matrix)
    y,x = np.where(mask >= 4)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    positions = np.concatenate((x,y),axis=1)
    
    return positions

def assign_direction(size):
    remainder = size%4
    number_direction = (size-remainder)/4
    list_li = list()
    for i in range(int(number_direction)):
        list_li.append((-1,0))
        list_li.append((1,0))
        list_li.append((0,-1))
        list_li.append((0,1))
    direction = np.array(list_li)
    np.random.shuffle(direction) 
    return direction

def create_mask(x,y):
    matrix = np.empty((y,x),dtype=object)
    matrix.fill(np.array([],dtype=int))
    return matrix 


def regulate_pile(matrix,iteration):
    location = find_pos(matrix)
    cascade_count = 0
    max_x = matrix.shape[1] - 1
    max_y = matrix.shape[0] - 1
    
    affected_area = []

    complete_history.append(matrix.copy())
    complete_history_index.append(iteration)

    falloff = []

    while len(location) != 0:
        mask = create_mask(matrix.shape[1],matrix.shape[0])
        cascade_count += len(location)

        for pos in location:
            container_size = (matrix[pos[1]][pos[0]]).shape[0]
            direction = assign_direction(container_size)

            for i in range(0,len(direction)):
                new_pos = direction[i] + pos
                if new_pos[1] > max_y or new_pos[1] < 0:
                    falloff.append((iteration,(new_pos[1],new_pos[0]),(matrix[pos[1]][pos[0]][container_size-1-i])))
                    continue
                elif new_pos[0] > max_x or new_pos[0]< 0:
                    falloff.append((iteration,(new_pos[1],new_pos[0]),(matrix[pos[1]][pos[0]][container_size-1-i])))
                    continue
                #print(mask[new_pos[1]][new_pos[0]])
                #print(matrix[new_pos[1]][new_pos[0]])
                #print((matrix[pos[1]][pos[0]][i]))

                mask[new_pos[1]][new_pos[0]] = np.append(mask[new_pos[1]][new_pos[0]],(matrix[pos[1]][pos[0]][container_size-1-i]))
            matrix[pos[1]][pos[0]] = matrix[pos[1]][pos[0]][0:container_size%4]

        vectorization = np.vectorize(lambda arr:arr.shape[0])
        vector_map = vectorization(mask)
        cascade_location_y,cascade_location_x = np.where(vector_map > 0)

        for i in range(len(cascade_location_y)):
            for j in range(len(mask[cascade_location_y[i]][cascade_location_x[i]])):
                if (cascade_location_y[i],cascade_location_x[i]) not in affected_area:
                    affected_area.append((cascade_location_y[i],cascade_location_x[i]))
                matrix[cascade_location_y[i]][cascade_location_x[i]] = np.append(matrix[cascade_location_y[i]][cascade_location_x[i]],mask[cascade_location_y[i]][cascade_location_x[i]][j])

        complete_history.append(matrix.copy())
        complete_history_index.append(iteration)
        location = find_pos(matrix)

    iteration_piles.append(matrix.copy())
    falloff = np.array(falloff).reshape(-1,3)
    fallout_edges.append(falloff)
    return cascade_count,(len(affected_area)/(matrix.shape[0]*matrix.shape[1]))*100


def track_sand(id):
    appeared= False
    data_entries_iteration = list()
    data_entries_location = list()
    for i,m in enumerate(complete_history):
        vectorization = np.vectorize(lambda x: id in x)
        feature_map = vectorization(m)
        exist = np.where(feature_map  == True)
        if exist[0].shape[0] > 0:
            appeared=True
            location = "("+str(exist[1][0]) + ","+ str(exist[0][0])+")"
            data_entries_iteration.append(str(complete_history_index[i]))
            data_entries_location.append(location)
            
        if  exist[0].shape[0] == 0 and appeared:
            break
    with open('sand_'+str(id)+'_track.csv', 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(["iteration","location"])
        for i,entry in enumerate(data_entries_location):
            filewriter.writerow([data_entries_iteration[i],data_entries_location[i]])
        #print(exist[0].shape,complete_history_index[i])

def get_file_name():
    counter = 0
    for file in glob.glob("*.png"):
        if file.startswith('sandpile_ver2.'):
            counter += 1
    new_filename = "sandpile_ver2." + str(counter+1)+".png"
    return new_filename

def redblack(arr,id):
    return np.where(arr==id)[0].shape[0]

def get_location(mat,id):
    vectorization_func = np.vectorize(redblack)
    feature_map = vectorization_func(mat,id)
    return feature_map

def getfilename(base):
    count = 0
    for file in glob.glob(base+"\\*.csv"):
        if "falloff_edges" in file:
            count += 1
    return "falloff_edges" +"_" +str(count) + ".csv"
def areafilename(base):
    count = 0
    for file in glob.glob(base+"\\*.csv"):
        if "cascade_area" in file:
            count += 1
    return "cascade_area" +"_" +str(count) + ".csv"

def black_red_queen_step(iteration,pile,red_prob):
    global stimulation
    global fallout_edges
    global current_model
    i = 0
    x = list()
    y = list()
    frequency_dict = dict()
    affected_history = list()

    for i in range(iteration):
        drop_sand_switch(pile,i,red_prob)
        cascade,area_affected = regulate_pile(pile,i)
        print("iteration: " +str(i)+ " number of cascade: "+ str(cascade))
        x.append(i)
        y.append(cascade)

        if cascade not in list(frequency_dict.keys()):
            frequency_dict[cascade] = 1
        else:
            frequency_dict[cascade] += 1

        affected_history.append(area_affected)
    current_model = str(time.time())
    np.savez(folder_name+"\\"+current_model,pile = pile)

    stimulation = stimulation+1

    filename = getfilename(folder_name)
    cascade_area = areafilename(folder_name)

    fallout_edges = fallout_edges
    fallout_edges = np.concatenate(fallout_edges)

    with open(folder_name+"/red_black_stimulation_"+str(stimulation)+"_"+filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration","position(y,x)","grain"])
        for i in fallout_edges:
            writer.writerow(i)
    with open(folder_name+"/red_black_stimulation_"+str(stimulation)+"_"+cascade_area, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration","area percentage"])
        for i in range(len(affected_history)):
            writer.writerow([i,int(affected_history[i])])

    fr_x = np.sort(np.array(list(frequency_dict.keys())),kind= 'mergesort')
    fr_y = []
    fr_x = fr_x[1:]

    for i in fr_x:
        fr_y.append(frequency_dict[i])
    fr_y = np.array(fr_y)
    fr_y = np.log10(fr_y)
    fr_x = np.log10(fr_x) 
    

    z= np.polyfit(fr_x,fr_y,deg=1)
    p = np.poly1d(z)
    dp = p(fr_x)
    equation, = ax.plot(fr_x,dp)
    equation_list.append(str(np.around(z[0],2))+"*x + "+ str(np.around(z[1],2)))
    equation_plot.append(equation)
    #plt.plot(fr_x,fr_y,c='g',label="Event Trending")
    #plt.scatter(fr_x,fr_y-dp,label ='detrend')
    scatter_plt = ax.scatter(fr_x,fr_y)
    scatter_plot.append(scatter_plt)
    scatter_list.append("stimulation: %d"%stimulation)


if __name__ == "__main__":
    with open("setup.csv") as f:
        csv_reader = csv.reader(f)
        csv_reader.__next__()
        for row in csv_reader:
            x = int(row[0])
            y = int(row[1])
            iteration = int(row[2])
            matrix = initialize_pile(x,y)
            black_red_queen_step(iteration,matrix,0)
            iteration_piles = []      
            fallout_edges = []

            pile = np.load(folder_name+"\\"+current_model+".npz",allow_pickle =True)["pile"]
            iteration_piles = []      
            fallout_edges = []

            black_red_queen_step(int(row[4]),pile,red_prob=float(row[3]))
            fallout_edges = []
            iteration_piles = np.array(iteration_piles)
            np.savez(folder_name+"\\"+current_model+"_piles.npz",allow_pickle =True,piles=iteration_piles)
            iteration_piles = []

            print(current_model)

        ax.set_xlabel("$log(Frequency)$")
        ax.set_ylabel("$log(magnitude)$")
        ax.set_title("log magnitude vs. frequency")
        ax.legend(equation_plot+scatter_plot,equation_list+scatter_list)
        plt.savefig("fat_tail.pdf")
        plt.savefig("fat_tail.png")

            

    #x = 30
    #y = 30
    #iteration = 8000

    #matrix = initialize_pile(x,y)
    #x = matrix.shape[1]
    #y = matrix.shape[0]
    #vectorization_local = np.vectorize(lambda arr:arr.shape[0])
    ##test = np.array([[np.array([]),np.array([]),np.array([])],[np.array([2,2,2,2]),np.array([1,1,1,1]),np.array([3,3,3,3])],[np.array([]),np.array([]),np.array([])]],dtype=object)
    ##regulate_pile(test)
    ##print(test)

    ##print(vectorization_local(test))
    #i = 0
    #x = list()
    #y = list()
    #frequency_dict = dict()

    #for i in range(iteration):
        #drop_sand(matrix,i)
        #cascade,area = regulate_pile(matrix,i)
        #print("iteration: " +str(i)+ " number of cascade: "+ str(cascade))
        #x.append(i)
        #y.append(cascade)

        #if cascade not in list(frequency_dict.keys()):
            #frequency_dict[cascade] = 1
        #else:
            #frequency_dict[cascade] += 1
    #plt.subplot(1,2,1)
    #plt.plot(np.array(x),np.array(y),label="Cascade Event")
    #plt.scatter(np.array(x),np.array(y),c='y',label="Cascade count")
    #plt.title("Object-based Search Graph")
    #plt.xlabel("Iteration Number")
    #plt.ylabel("Casade Number")
    #plt.legend()

    #plt.subplot(1,2,2)
    #fr_x = np.sort(np.array(list(frequency_dict.keys())),kind= 'mergesort')
    #fr_y = []
    #fr_x = fr_x[1:]
    #for i in fr_x:
        #fr_y.append(frequency_dict[i])
    #fr_y = np.array(fr_y)
    #fr_y = np.log10(fr_y)
    #fr_x = np.log10(fr_x) 


    #z= np.polyfit(fr_x,fr_y,deg=1)
    #print(z)
    #p = np.poly1d(z)
    #dp = p(fr_x)

    #plt.plot(fr_x,dp,label='fit')
    #plt.plot(fr_x,fr_y,c='g',label="Event Trending")
    #plt.scatter(fr_x,fr_y-dp,label ='detrend')
    #plt.scatter(fr_x,fr_y,c ='r',label="occurance count")
    #plt.xlabel("Number of Casacade log(10)")
    #plt.ylabel("Frequency for given occurance (log10)")
    #plt.title(str(np.around(z[0],2))+"*x + "+ str(np.around(z[1],2)))
    #plt.legend()
    #plt.show()

    #red = get_location(matrix,1)
    #black = get_location(matrix,0)
    #v_func = np.vectorize(lambda arr: arr.shape[0])
    #temp_arr = v_func(matrix)


    #red_percentage = np.sum(red)/ np.sum(temp_arr)
    #black_percentage = np.sum(black)/np.sum(temp_arr)

    #temp_arr = v_func(matrix)

    #equal_val = red + black
    #fallout_edges = np.concatenate(fallout_edges)


    #with open(getfilename("18.05.59"), 'w', newline='') as csvfile:
        #writer = csv.writer(csvfile)
        #writer.writerow(["iteration","position(y,x)","grain"])
        #for i in fallout_edges:
            #writer.writerow(i)