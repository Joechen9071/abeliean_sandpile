import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy, linregress


complete_history = []
complete_history_index = []
fallout_edges = []
elite_cap = 2**5
non_elite_cap = 2**4
matrix = None
masking = None
area_volume_heatmap = None
area_events_heatmap = None
msg_cipher = ["a", "b", "c", "d", "e", "f", "g",
              "h", "i", "j", "k", "l", "m", "n", "o", "p"]
msg_decipher = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

stop = False


def encoder_msg(msg):
    '''
    function take msg as string then cipher to the integer than transform into n-bits binary string
    '''
    idx = msg_cipher.index(msg)
    cipher = msg_decipher[idx]
    cipher_binary = format(cipher, '04b')
    return cipher_binary


def decoder_msg(msg):
    integer = int(msg, 2)
    idx = msg_decipher.index(integer)
    reconstructed_msg = msg_cipher[idx]
    return reconstructed_msg


def initliaze_elite(x, y, sx, sy):
    global masking
    if sx % 2 == 0 and sy % 2 == 0:
        if sx != 0 and sy != 0:
            raise AssertionError("elite size must be odd")
    masking = np.zeros((x, y), dtype=np.int8)
    center = masking.shape[0]//2
    for i in range(center-(sy//2), center+(sy//2)+1):
        if sx == 0 and sy == 0:
            break
        masking[i][center-sx//2:center+sx//2+1] = 1


def initialize_pile(x, y, elite_size):
    global matrix, area_volume_heatmap, area_events_heatmap
    '''
    
    \Parameters\: Define size by given x and y 
    x: columns of matrix 
    y: height of matrix 

    output: empty matrix with shape (y,x,4) with -1 filled inside
    '''
    matrix = np.empty((y, x), dtype=object)
    matrix.fill(np.array([], dtype=int))
    initliaze_elite(x, y, elite_size[0], elite_size[1])

    area_volume_heatmap = np.zeros((y, x), dtype=np.int64)
    area_events_heatmap = np.zeros((y, x), dtype=np.int64)

    return matrix


def drop_sand(pile_container, msg):
    '''
    Paramters: matrix the sand will be drop in and id for current drop

    pile_container: expected matrix to be shape of (y,x,4)
    id: int represents sand id 

    output: updated the sand pile with id updated.
    '''
    x = pile_container.shape[1]
    y = pile_container.shape[0]
    location = (np.random.randint(0, y, (1,)).flatten()[
                0], np.random.randint(0, x, (1,)).flatten()[0])
    current_container = pile_container[location[0]][location[1]]
    msg = encoder_msg(msg)
    for i in range(len(msg)):
        current_container = np.append(current_container, msg[i])
    pile_container[location[0]][location[1]] = current_container

    return pile_container


def find_pos(matrix):
    positions = []
    vector_x = np.vectorize(lambda arr: arr.shape[0])
    mask = vector_x(matrix)
    # print(mask)
    # print(np.sum(mask))
    y, x = np.where(mask >= non_elite_cap)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    positions = []
    for i in range(x.shape[0]):
        if masking[y[i][0]][x[i][0]] == 0 and mask[y[i][0]][x[i][0]] >= non_elite_cap:
            positions.append([x[i][0], y[i][0]])
        elif masking[y[i][0]][x[i][0]] == 1 and mask[y[i][0]][x[i][0]] >= elite_cap:
            positions.append([x[i][0], y[i][0]])

    positions = np.array(positions)

    return positions


def preview_directions():
    list_li = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return list_li


def assign_direction_elite(size, pos, cascade_elite):
    if masking[pos[1]][pos[0]] == 0:
        if cascade_elite:
            list_li = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            direction = np.array(list_li)
            elite_direction = []
            non_elite_direction = []

            for i in list_li:
                location = pos+i
                if masking[location[1]][location[0]] == 1:
                    elite_direction.append(i)
                else:
                    non_elite_direction.append(i)
            non_elite_direction = ((size - (3*4))//3)*non_elite_direction
            direction = np.array(non_elite_direction + elite_direction)

            np.random.shuffle(direction)
        else:
            remainder = size % 4
            number_direction = (size-remainder)/4
            list_li = list()
            for i in range(int(number_direction)):
                list_li.append((-1, 0))
                list_li.append((1, 0))
                list_li.append((0, -1))
                list_li.append((0, 1))
            direction = np.array(list_li)
            np.random.shuffle(direction)
    else:
        remainder = size % 4
        number_direction = (size-remainder)/4
        list_li = list()
        for i in range(int(number_direction)):
            list_li.append((-1, 0))
            list_li.append((1, 0))
            list_li.append((0, -1))
            list_li.append((0, 1))
        additional_moves = np.array(preview_directions())
        np.random.shuffle(additional_moves)
        additional_moves = additional_moves[0:remainder]
        direction = np.array(list_li)
        direction = np.vstack([direction, additional_moves])
        np.random.shuffle(direction)
    return direction


def create_mask(x, y):
    matrix = np.empty((y, x), dtype=object)
    matrix.fill(np.array([], dtype=np.int32))
    return matrix


def regulate_pile(matrix, iteration, trigger=None):
    location = find_pos(matrix)
    cascade_count = 0
    cascade_magnitude = 0
    max_x = matrix.shape[1] - 1
    max_y = matrix.shape[0] - 1

    affected_area = []

    complete_history.append(matrix.copy())
    complete_history_index.append(iteration)

    falloff = []
    vectorization = np.vectorize(lambda arr: arr.shape[0])

    while len(location) != 0:

        mask = create_mask(matrix.shape[1], matrix.shape[0])
        cascade_count += len(location)

        for pos in location:
            pile_pointer = (matrix[pos[1]][pos[0]]).shape[0]
            original_container_size = (matrix[pos[1]][pos[0]]).shape[0]
            direction = preview_directions()

            cascade_elite = False
            neighbor = False
            via_location = []
            prob = []
            for j in direction:
                temp = j + pos
                if temp[1] > max_y or temp[1] < 0:
                    continue
                elif temp[0] > max_x or temp[0] < 0:
                    continue
                if masking[temp[1]][temp[0]] == 1 and masking[pos[1]][pos[0]] == 0:
                    if pile_pointer > 5*4:
                        cascade_elite = True
                        neighbor = True
                    else:
                        neighbor = True
                        continue

            direction = assign_direction_elite(
                original_container_size, pos, cascade_elite)

            for j in direction:
                temp = j + pos
                if temp[1] > max_y or temp[1] < 0:
                    via_location.append(temp)
                    prob.append(np.random.random())
                    continue
                elif temp[0] > max_x or temp[0] < 0:
                    via_location.append(temp)
                    prob.append(np.random.random())
                    continue
                if masking[temp[1]][temp[0]] == 1 and masking[pos[1]][pos[0]] == 0 and pile_pointer > 5*4:
                    via_location.append(temp)
                elif masking[temp[1]][temp[0]] == 1 and masking[pos[1]][pos[0]] == 0 and pile_pointer <= 5*4:
                    continue
                else:
                    via_location.append(temp)
                    prob.append(np.random.random())

            max_prob_idx = np.argmax(prob)
            max_prob_indices = np.zeros(len(via_location))
            for i, obj in enumerate(via_location):
                if (via_location[max_prob_idx] == via_location[i]).all():
                    max_prob_indices[i] = 1

            if cascade_elite and neighbor:
                max_prob_indices = np.zeros(len(via_location))
            if not neighbor:
                max_prob_indices = np.zeros(len(via_location))

            for i, new_pos in enumerate(via_location):
                if new_pos[1] > max_y or new_pos[1] < 0:
                    falloff.append(
                        (iteration, (new_pos[1], new_pos[0]), (matrix[pos[1]][pos[0]][pile_pointer-1-i])))
                    if max_prob_indices[i] == 1:
                        pile_pointer -= 1
                    continue
                elif new_pos[0] > max_x or new_pos[0] < 0:
                    falloff.append(
                        (iteration, (new_pos[1], new_pos[0]), (matrix[pos[1]][pos[0]][pile_pointer-1-i])))
                    if max_prob_indices[i] == 1:
                        pile_pointer -= 1
                    continue
                if masking[new_pos[1]][new_pos[0]] == 1 and masking[pos[1]][pos[0]] == 0:
                    if pile_pointer > 4*3:
                        bits_to_fall = original_container_size-(4*3)
                        mask[new_pos[1]][new_pos[0]] = np.append(mask[new_pos[1]][new_pos[0]], (
                            matrix[pos[1]][pos[0]][pile_pointer-i-bits_to_fall:pile_pointer-i]))
                        pile_pointer -= bits_to_fall-1
                    else:
                        continue
                else:
                    # print(max_prob_idx,new_pos,pile_pointer)
                    if max_prob_indices[i] == 1:
                        # print(matrix[pos[1]][pos[0]][pile_pointer-1-i-1:pile_pointer-i])
                        mask[new_pos[1]][new_pos[0]] = np.append(
                            mask[new_pos[1]][new_pos[0]], (matrix[pos[1]][pos[0]][pile_pointer-1-i-1:pile_pointer-i]))
                        pile_pointer -= 1
                    else:
                        mask[new_pos[1]][new_pos[0]] = np.append(
                            mask[new_pos[1]][new_pos[0]], (matrix[pos[1]][pos[0]][pile_pointer-1-i]))
            if cascade_elite:
                matrix[pos[1]][pos[0]] = np.array([], dtype=np.int32)
            elif masking[pos[1]][pos[0]] == 1:
                matrix[pos[1]][pos[0]] = np.array([], dtype=np.int32)
            else:
                matrix[pos[1]][pos[0]] = matrix[pos[1]
                                                ][pos[0]][0:original_container_size % 4]

        vector_map = vectorization(mask)
        cascade_magnitude += np.sum(vector_map)
        cascade_location_y, cascade_location_x = np.where(vector_map > 0)
        for i in range(len(cascade_location_y)):
            if (cascade_location_y[i], cascade_location_x[i]) not in affected_area:
                affected_area.append(
                    (cascade_location_y[i], cascade_location_x[i]))
            matrix[cascade_location_y[i]][cascade_location_x[i]] = np.append(
                matrix[cascade_location_y[i]][cascade_location_x[i]], mask[cascade_location_y[i]][cascade_location_x[i]])

            area_events_heatmap[cascade_location_y[i]
                                ][cascade_location_x[i]] += 1
            area_volume_heatmap[cascade_location_y[i]][cascade_location_x[i]
                                                       ] += len(matrix[cascade_location_y[i]][cascade_location_x[i]])

        complete_history.append(matrix.copy())
        complete_history_index.append(iteration)
        location = find_pos(matrix)

    falloff = np.array(falloff, dtype=object).reshape(-1, 3)
    fallout_edges.append(falloff)
    return cascade_count, (len(affected_area)/(matrix.shape[0]*matrix.shape[1]))*100, cascade_magnitude, vectorization(matrix)


def track_sand(id):
    appeared = False
    data_entries_iteration = list()
    data_entries_location = list()
    for i, m in enumerate(complete_history):
        vectorization = np.vectorize(lambda x: id in x)
        feature_map = vectorization(m)
        exist = np.where(feature_map == True)
        if exist[0].shape[0] > 0:
            appeared = True
            location = "("+str(exist[1][0]) + "," + str(exist[0][0])+")"
            data_entries_iteration.append(str(complete_history_index[i]))
            data_entries_location.append(location)

        if exist[0].shape[0] == 0 and appeared:
            break
    with open('sand_'+str(id)+'_track.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(["iteration", "location"])
        for i, entry in enumerate(data_entries_location):
            filewriter.writerow(
                [data_entries_iteration[i], data_entries_location[i]])
        # print(exist[0].shape,complete_history_index[i])


def get_file_name():
    counter = 0
    for file in glob.glob("*.png"):
        if file.startswith('sandpile_ver2.'):
            counter += 1
    new_filename = "sandpile_ver2." + str(counter+1)+".png"
    return new_filename


def redblack(arr, id):
    return np.where(arr == id)[0].shape[0]


def get_location(mat, id):
    vectorization_func = np.vectorize(redblack)
    feature_map = vectorization_func(mat, id)
    return feature_map


def getfilename(base):
    count = 0
    for file in glob.glob(base+"\\*.csv"):
        if "falloff_edges" in file:
            count += 1
    return "falloff_edges" + "_" + str(count) + ".csv"


def areafilename(base):
    count = 0
    for file in glob.glob(base+"\\*.csv"):
        if "cascade_area" in file:
            count += 1
    return "cascade_area" + "_" + str(count) + ".csv"


def update_information(record, grid):
    for j in range(grid.shape[0]):
        for i in range(grid.shape[1]):
            cell = grid[j][i]
            complete_information = cell.shape[0]//4
            last_information = cell.shape[0] % 4
            if complete_information == 0 and last_information == 0:
                continue
            else:
                if not last_information == 0:
                    last_code = cell[(-1*last_information):]
                    previous_code = cell[:(-1*last_information)]
                    for p in range(complete_information):
                        bin_code = ''.join(list(previous_code[p*4:((p+1)*4)]))
                        msg = decoder_msg(bin_code)
                        record[msg] = record[msg]+1
                    bin_code = ''.join(list(last_code))
                    msg = decoder_msg(bin_code)
                    record[msg] = record[msg] + 1
                else:
                    previous_code = cell
                    for p in range(complete_information):
                        bin_code = ''.join(list(previous_code[p*4:((p+1)*4)]))
                        msg = decoder_msg(bin_code)
                        record[msg] = record[msg]+1

            # print(complete_information,last_information)


def entropy_loss(actual, prediction):
    Y = np.array(actual, dtype=np.float64)
    P = np.array(prediction, dtype=np.float64)

    CE = -1*np.sum(Y*np.log(P))
    return entropy(Y, P, base=26)
# def plot_references(results,note,ylabel,overlap=False,ref_index=-1,to_sort=True):
    # if overlap:
    # fig,axes = plt.subplots(2,(len(results)-1)//2)
    # else:
    # fig,axes = plt.subplots(len(results)-1,2)
    # for config in range(len(results)-1):
    # cas = results[config]
    # ref = results[-1]
    # if to_sort:
    # cas.sort()
    # ref.sort()
    # cas = np.array(cas)
    # ref = np.array(ref)
#
    # print(len(cas))
    # cas = (cas - np.min(cas))/(np.max(cas)-np.min(cas))
    # ref = (ref - np.min(ref))/(np.max(ref)-np.min(ref))
    # if overlap:
    # if config%2 != 0:
    # temp = 1
    # else:
    # temp = 0
    # vertical_idx = config//2
    # axes[vertical_idx][temp].plot(range(0,len(cas)),cas+1e-5,c=colors[config],linestyle="-.")
    # axes[vertical_idx][temp].set_xlabel('# of iteration')
    # axes[vertical_idx][temp].set_ylabel(ylabel)
#
    # axes[vertical_idx][temp].plot(range(0,len(cas)),ref+1e-5,c=colors[-1],linestyle='--')
    # axes[vertical_idx][temp].set_xlabel('# of iteration')
    # axes[vertical_idx][temp].set_ylabel(ylabel)
    # axes[vertical_idx][temp].legend(["%s %d*%d"%(note,area_size[config],area_size[config]),"%s %d*%d"%(note,area_size[-1],area_size[-1])])
    # else:
    # fig_ind = -1
    # axes[config][0].plot(range(len(cas)),cas,c=colors[config])
    # axes[config][0].set_xlabel('# of iteration')
    # axes[config][0].set_ylabel('cascade events')
    # axes[config][0].legend(["area size %d*%d"%(area_size[config],area_size[config])])
#
    # axes[config][fig_ind].plot(range(len(cas)),ref,c=colors[-1])
    # axes[config][fig_ind].set_xlabel('# of iteration')
    # axes[config][fig_ind].set_ylabel('cascade events')
    # axes[config][fig_ind].legend(["area size %d*%d"%(area_size[-1],area_size[-1])])
#
    # plt.tight_layout()
    # plt.show()


def fat_tail_curve(frequency_dict):
    fr_x = np.sort(np.array(list(frequency_dict.keys())), kind='mergesort')
    fr_y = []
    fr_x = fr_x[1:]
    dp = []

    if len(fr_x) > 0:
        for i in fr_x:
            fr_y.append(frequency_dict[i])

        fr_y = np.array(fr_y)
        fr_y = np.log10(fr_y)

        valid_indices = np.where(fr_y != 0)
        fr_x = fr_x[valid_indices]
        fr_y = fr_y[valid_indices]

        return np.log10(fr_x), fr_y, None

    return np.log10(fr_x), fr_y, np.zeros(np.array(fr_y).shape)


if __name__ == "__main__":
    if not os.path.exists("configuration\\capacity%d" % elite_cap):
        os.mkdir("configuration\\capacity%d" % elite_cap)
    export_folder = "configuration\\capacity%d" % elite_cap

    x_config = [25, 25, 25, 25, 25]
    y_config = [25, 25, 25, 25, 25]
    iterations = [2000]*5
    area_size = [15, 11, 7, 3, 0]
    colors = ["red", "blue", "orange", "purple", "green"]
    linestyle_tuple = [
        ('dotted',                (0, (1, 1))),
        ('densely dotted',        (0, (1, 1))),
        ('long dash with offset', (5, (10, 3))),
        ('dashed',                (0, (5, 5))),
        ('densely dashed',        (0, (5, 1)))]
    cas_res = []
    area_res = []
    vol_res = []
    volume_heatmaps = []
    events_heatmaps = []
    loss_result = []
    for config in range(len(iterations)):
        iteration = iterations[config]
        matrix = initialize_pile(
            x_config[config], y_config[config], (area_size[config], area_size[config]))
        print(masking)
        # print(np.where(masking == 1))

        frequency_dict = dict()

        information_transmitted = {
            "a": 1e-3,
            "b": 1e-3,
            "c": 1e-3,
            "d": 1e-3,
            "e": 1e-3,
            "f": 1e-3,
            "g": 1e-3,
            "h": 1e-3,
            "i": 1e-3,
            "j": 1e-3,
            "k": 1e-3,
            "l": 1e-3,
            "m": 1e-3,
            "n": 1e-3,
            "o": 1e-3,
            "p": 1e-3
        }
        information_retained = {
            "a": 1e-3,
            "b": 1e-3,
            "c": 1e-3,
            "d": 1e-3,
            "e": 1e-3,
            "f": 1e-3,
            "g": 1e-3,
            "h": 1e-3,
            "i": 1e-3,
            "j": 1e-3,
            "k": 1e-3,
            "l": 1e-3,
            "m": 1e-3,
            "n": 1e-3,
            "o": 1e-3,
            "p": 1e-3
        }
        cas = []
        area = []
        volume = []

        loss_history = []

        for i in range(iteration):

            msg_sand = np.random.choice(msg_cipher, size=1)[0]
            information_transmitted[msg_sand] = information_transmitted[msg_sand] + 1
            matrix = drop_sand(matrix, msg_sand)
            cas_count, area_affected, cascade_magnitude, matrix_mag = regulate_pile(
                matrix, i)

            update_information(information_retained, matrix)
            ce = entropy_loss(list(information_transmitted.values()), list(
                information_retained.values()))

            loss_history.append(ce)
            cas.append(cas_count)
            area.append(area_affected)
            volume.append(cascade_magnitude)
            if cascade_magnitude not in list(frequency_dict.keys()):
                frequency_dict[cascade_magnitude] = 1
            else:
                frequency_dict[cascade_magnitude] += 1

            # print(i)
            if (i+1) % 1000 == 0:
                print("iteration: #%d completed" % (i+1))
        # print(loss_history)
        loss_result.append(loss_history)
        # store heatmaps
        vectorization = np.vectorize(lambda arr: arr.shape[0])
        final_size = vectorization(matrix)
        # print(np.sum(final_size))
        volume_heatmaps.append(np.copy(area_volume_heatmap))
        events_heatmaps.append(np.copy(area_events_heatmap))

        # start plotting fat-tail graph
        cas_res.append(cas)
        area_res.append(area)
        vol_res.append(volume)

        fr_x = np.sort(np.array(list(frequency_dict.keys())), kind='mergesort')
        fr_y = []
        fr_x = fr_x[1:]

        for i in fr_x:
            fr_y.append(frequency_dict[i])
        fr_y = np.array(fr_y)
        fr_y = np.log10(fr_y)
        fr_x = np.log10(fr_x)

        z = np.polyfit(fr_x, fr_y, deg=1)
        p = np.poly1d(z)
        dp = np.exp(p(fr_x))
        print(fr_x.shape, dp.shape)
        plt.plot(fr_x, dp, label=str(np.around(
            z[0], 2))+"*x + " + str(np.around(z[1], 2)) + " size %d" % area_size[config])
        plt.title("Frequency Graph with Different Elite Area")
        plt.scatter(fr_x, fr_y)
        plt.legend()
        plt.savefig(export_folder+"\\fat-tail.png")

    plt.clf()
    # for idx, vmap in enumerate(volume_heatmaps):
    # plt.imshow(vmap)
    # plt.colorbar()
    # plt.savefig("configuration\\%s_area_heatmap.png"%area_size[idx],bbox_inches='tight')
    # plt.clf()
    for i, s in enumerate(area_size):
        plt.plot(loss_result[i],
                 label="entropy loss for each area size %d" % s)
        plt.legend()
    plt.savefig(export_folder + "\\entropy_loss.png")

    for idx, (vmap, emap) in enumerate(zip(volume_heatmaps, events_heatmaps)):
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        divider = make_axes_locatable(axes[0])
        colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)

        axes[0].set_title("magnitude \n area %d capacity %d" %
                          (area_size[idx], elite_cap))
        axes[0].title.set_size(20)
        img1 = axes[0].imshow(vmap, cmap="jet", interpolation='nearest')
        plt.colorbar(img1, colorbar_axes)

        divider = make_axes_locatable(axes[1])
        colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
        axes[1].set_title("events \n area %d capacity %d" %
                          (area_size[idx], elite_cap))
        axes[1].title.set_size(20)
        img2 = axes[1].imshow(emap, interpolation='nearest')
        plt.colorbar(img2, colorbar_axes)

        plt.savefig(export_folder+"\\%s_area_heatmap.png" %
                    area_size[idx], bbox_inches='tight')
        plt.clf()

    plt_titles = ["Cascade events",
                  "percent of area affected", "information bits moved"]
    plt_series = [cas_res, area_res, vol_res]
    for i in range(3):
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        for j in range(len(plt_series[i])):
            graph = plt_series[i][j]
            graph.sort()
            axes.plot(range(len(graph)), np.array(graph), label="area %d*%d" %
                      (area_size[j], area_size[j]), linestyle=linestyle_tuple[j][1])
        axes.set_title(plt_titles[i])
        axes.legend()
        plt.savefig(export_folder+"\\%s.png" %
                    plt_titles[i], bbox_inches='tight')
        plt.clf()
    # plot_references(cas_res,note="area size",ylabel="cascade events",overlap=True,to_sort=False)
    # plot_references(area_res,note="area size",ylabel="area affected (%)",overlap=True,to_sort=False)
    # plot_references(vol_res,note="area size",ylabel="number of bits moved",overlap=True,to_sort=False)

    # information_retained = {
        # "a":0,
        # "b":0,
        # "c":0,
        # "d":0,
        # "e":0,
        # "f":0,
        # "g":0,
        # "h":0,
        # "i":0,
        # "j":0,
        # "k":0,
        # "l":0,
        # "m":0,
        # "n":0,
        # "o":0,
        # "p":0
    # }
    # pile = initialize_pile(5,5,(1,1))
    # pile[2][1] = np.array(['1','0','1','1',
        # '0','0','0','0',
        # '0','0','0','0',
        # '0','0','0','0',
        # '0','0','0','0',
        # '0','0','0','0'])
    # matrix = np.array([[np.array([]),np.array([]),np.array([])],
        # [np.array([]),np.array([]),np.array([1,2,3,4,5,6])],
        # [np.array([]),np.array([]),np.array([])]],dtype=object)
    # update_information(information_retained,pile)
    # regulate_pile(pile,0)
    # print(pile)
    # print(information_retained)
