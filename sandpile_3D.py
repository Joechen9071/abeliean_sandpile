import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import csv


complete_history = []
complete_history_index = []
fallout_edges= []
matrix = None
masking = None
def initliaze_elite(x,y,sx,sy):
    global masking
    if sx%2 == 0 and sy%2 == 0 :
        if sx != 0 and sy!=0:
            raise AssertionError("elite size must be odd")
    masking = np.zeros((x,y),dtype=np.int8)
    center = masking.shape[0]//2
    for i in range(center-(sy//2),center+(sy//2)+1):
        if sx == 0 and sy == 0:
            break
        masking[i][center-sx//2:center+sx//2+1] = 1


def initialize_pile(x,y,elite_size):
    global matrix
    '''
    
    \Parameters\: Define size by given x and y 
    x: columns of matrix 
    y: height of matrix 

    output: empty matrix with shape (y,x,4) with -1 filled inside
    '''
    matrix = np.empty((y,x),dtype=object)
    matrix.fill(np.array([],dtype=int))
    initliaze_elite(x,y,elite_size[0],elite_size[1])
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
def assign_direction_elite(size,pos):
    #if size > 5:
        #print(masking[pos[1]][pos[0]],matrix[pos[1]][pos[0]],pos)
        
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
    matrix.fill(np.array([],dtype=np.int32))
    return matrix 


def regulate_pile(matrix,iteration):
    location = find_pos(matrix)
    cascade_count = 0
    cascade_magnitude = 0
    max_x = matrix.shape[1] - 1
    max_y = matrix.shape[0] - 1

    affected_area = []

    complete_history.append(matrix.copy())
    complete_history_index.append(iteration)

    falloff = []

    while len(location) != 0:
        #print(matrix)
        mask = create_mask(matrix.shape[1],matrix.shape[0])
        cascade_count += len(location)

        #print(len(location))
        for pos in location:
            
            pile_pointer = (matrix[pos[1]][pos[0]]).shape[0]
            original_container_size = (matrix[pos[1]][pos[0]]).shape[0]
            direction = assign_direction_elite(original_container_size,pos)
            cascade_elite = False
            neighbor = False
            via_location = []
            prob = []
            for j in direction:
                temp = j + pos
                if temp[1] > max_y or temp[1] < 0:
                    via_location.append(temp)
                    prob.append(np.random.random())
                    continue
                elif temp[0] > max_x or temp[0]< 0:
                    via_location.append(temp)
                    prob.append(np.random.random())
                    continue

                if masking[temp[1]][temp[0]] == 1 and masking[pos[1]][pos[0]]==0 and pile_pointer > 5:
                    cascade_elite = True 
                    neighbor = True
                    via_location.append(temp)
                elif masking[temp[1]][temp[0]] == 1 and masking[pos[1]][pos[0]]==0 and pile_pointer < 5:
                    neighbor = True
                    continue
                else:
                    via_location.append(temp)
                    prob.append(np.random.random())
            max_prob_idx = np.argmax(prob)

            if cascade_elite and neighbor:
                max_prob_idx = -1
            if not neighbor:
                max_prob_idx = -1

            for i,new_pos in enumerate(via_location):
                if new_pos[1] > max_y or new_pos[1] < 0:
                    falloff.append((iteration,(new_pos[1],new_pos[0]),(matrix[pos[1]][pos[0]][pile_pointer-1-i])))
                    if max_prob_idx == i:
                        pile_pointer -=1
                    continue
                elif new_pos[0] > max_x or new_pos[0]< 0:
                    falloff.append((iteration,(new_pos[1],new_pos[0]),(matrix[pos[1]][pos[0]][pile_pointer-1-i])))
                    if max_prob_idx == i:
                        pile_pointer -=1
                    continue
                if masking[new_pos[1]][new_pos[0]] == 1 and masking[pos[1]][pos[0]]==0 and pile_pointer > 5:
                    mask[new_pos[1]][new_pos[0]] = np.append(mask[new_pos[1]][new_pos[0]],(matrix[pos[1]][pos[0]][pile_pointer-i-3:pile_pointer-i]))
                    pile_pointer -= 2
                elif masking[new_pos[1]][new_pos[0]] == 1 and masking[pos[1]][pos[0]]==0 and pile_pointer < 5:
                    continue
                else:
                    #print(max_prob_idx,new_pos,pile_pointer)
                    if max_prob_idx == i:
                        #print(matrix[pos[1]][pos[0]][pile_pointer-1-i-1:pile_pointer-i])
                        mask[new_pos[1]][new_pos[0]] = np.append(mask[new_pos[1]][new_pos[0]],(matrix[pos[1]][pos[0]][pile_pointer-1-i-1:pile_pointer-i]))
                        pile_pointer -= 1
                    else:
                        mask[new_pos[1]][new_pos[0]] = np.append(mask[new_pos[1]][new_pos[0]],(matrix[pos[1]][pos[0]][pile_pointer-1-i]))

            if cascade_elite:
                matrix[pos[1]][pos[0]] = matrix[pos[1]][pos[0]][0:original_container_size%6]
            else:
                matrix[pos[1]][pos[0]] = matrix[pos[1]][pos[0]][0:original_container_size%4]

        vectorization = np.vectorize(lambda arr:arr.shape[0])
        vector_map = vectorization(mask)
        cascade_magnitude += np.sum(vector_map)
        cascade_location_y,cascade_location_x = np.where(vector_map > 0)

        for i in range(len(cascade_location_y)):
            for j in range(len(mask[cascade_location_y[i]][cascade_location_x[i]])):
                if (cascade_location_y[i],cascade_location_x[i]) not in affected_area:
                    affected_area.append((cascade_location_y[i],cascade_location_x[i]))
                matrix[cascade_location_y[i]][cascade_location_x[i]] = np.append(matrix[cascade_location_y[i]][cascade_location_x[i]],mask[cascade_location_y[i]][cascade_location_x[i]][j])
        #print(matrix)
        complete_history.append(matrix.copy())
        complete_history_index.append(iteration)
        location = find_pos(matrix)

    falloff = np.array(falloff,dtype=object).reshape(-1,3)
    fallout_edges.append(falloff)
    return cascade_count,(len(affected_area)/(matrix.shape[0]*matrix.shape[1]))*100,cascade_magnitude


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



if __name__ == "__main__":
    x_config = [25,25,25,25,25]
    y_config = [25,25,25,25,25]
    iterations = [10000]*5
    area_size = [7,5,3,0]
    colors = ["red","blue","orange","green","black"]

    cas_res = []
    area_res = []
    vol_res = []
    for config in range(len(iterations)):
        iteration = iterations[config]
        matrix = initialize_pile(x_config[config],y_config[config],(area_size[config],area_size[config]))
        print(masking)
        cas = []
        area = []
        volume = []
        for i in range(iteration):
            matrix = drop_sand(matrix,0)
            cas_count,area_affected,cascade_magnitude = regulate_pile(matrix,i)
            cas.append(cas_count)
            area.append(area_affected)
            volume.append(cascade_magnitude)
            if (i+1)%1000 == 0:
                print("iteration: #%d completed"%(i+1))
        cas_res.append(cas)
        area_res.append(area)
        vol_res.append(volume)

    fig,axes = plt.subplots(len(iterations))
    for config in range(5):
        cas = cas_res[config]
        axes[config].plot(range(len(cas)),cas,c=colors[config])
        axes[config].set_xlabel('# of iteration')
        axes[config].set_ylabel('cascade events')
        axes[config].legend(["area size %d*%d"%(area_size[config],area_size[config])])
    plt.show()
    #plt.clf()

    fig,axes = plt.subplots(len(iterations))
    for config in range(3):
        area = area_res[config]
        axes[config].plot(range(len(area)),area,c=colors[config])
        axes[config].set_xlabel('# of iteration')
        axes[config].set_ylabel('area affected (%)')
        axes[config].legend(["area size %d*%d"%(area_size[config],area_size[config])])
    plt.show()

    fig,axes = plt.subplots(len(iterations))
    for config in range(5):
        mag = vol_res[config]
        axes[config].plot(range(len(area)),mag,c=colors[config])
        axes[config].set_xlabel('# of iteration')
        axes[config].set_ylabel('magnitude')
        axes[config].legend(["area size %d*%d"%(area_size[config],area_size[config])])
    plt.show()
    #matrix = np.array([[np.array([]),np.array([]),np.array([])],
             #[np.array([]),np.array([]),np.array([1,2,3,4,5,6])],
             #[np.array([]),np.array([]),np.array([])]],dtype=object)
    #
    #regulate_pile(matrix,0)

