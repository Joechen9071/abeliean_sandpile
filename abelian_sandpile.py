import numpy as np 
import matplotlib.pyplot as plt

pile = np.array(np.zeros((30,30),dtype=np.float32))

def regulate_pile(pile):
    cascade = 0
    k = np.zeros(pile.shape)
    k[:] = 255
    frame= np.multiply(pile/4,k)
    frame = np.around(frame)
    '''cv2.imshow("pile",cv2.resize(frame, (1000,1000), interpolation = cv2.INTER_AREA ))
    cv2.waitKey(1)'''
    while len(np.argwhere(pile>=4)) > 0:
        cascade += len(np.argwhere(pile>=4))
        indices = np.argwhere(pile>=4)
        mask = np.zeros(pile.shape)
        after_pile = np.copy(pile)
        for i in indices:
            y = i[0]
            x = i[1]
            if x+1 < pile.shape[1]:
                mask[y][x+1] += (pile[y][x] - (pile[y][x]%4))/4
            if x-1 > -1:
                mask[y][x-1] += (pile[y][x] - (pile[y][x]%4))/4
            if y + 1 < pile.shape[0]:
                mask[y+1][x] += (pile[y][x] - (pile[y][x]%4))/4
            if y-1 > -1:
                mask[y-1][x] += (pile[y][x] - (pile[y][x]%4))/4
            after_pile[y][x] = pile[y][x]%4
        after_pile = after_pile + mask
        pile = np.copy(after_pile)

        k = np.zeros(pile.shape)
        k[:] = 255
        frame= np.multiply(pile/4,k)
        frame = np.around(frame)
        ''' cv2.imshow("pile",cv2.resize(frame, (1000,1000), interpolation = cv2.INTER_AREA))
        cv2.waitKey(1)'''

    return pile,cascade

def drop_sand(pile):
    location = np.random.randint([pile.shape[0],pile.shape[1]],size=2)
    drop_y = location[0]
    drop_x = location[1]
    pile[drop_y][drop_x] += 1

    return pile

'''def regulate_pile():
    global pile
    cascade = 0
    while len(np.argwhere(pile>=4)) > 0:
        cascade += len(np.argwhere(pile>=4))
        indices = np.argwhere(pile>=4)
        mask = np.zeros(pile.shape)
        after_pile = np.copy(pile)
        for i in indices:
            y = i[0]
            x = i[1]
            if x+1 < pile.shape[1]:
                mask[y][x+1] += (pile[y][x] - (pile[y][x]%4))/4
            if x-1 > -1:
                mask[y][x-1] += (pile[y][x] - (pile[y][x]%4))/4
            if y + 1 < pile.shape[0]:
                mask[y+1][x] += (pile[y][x] - (pile[y][x]%4))/4
            if y-1 > -1:
                mask[y-1][x] += (pile[y][x] - (pile[y][x]%4))/4
            after_pile[y][x] = pile[y][x]%4
        after_pile = after_pile + mask
        pile = np.copy(after_pile)
    return pile,cascade

def drop_sand():
    location = np.random.randint([pile.shape[0],pile.shape[1]],size=2)
    drop_y = location[0]
    drop_x = location[1]
    pile[drop_y][drop_x] += 1
    return pile

def update(i):
    pile = np.copy(drop_sand())
    regulate,cascade = regulate_pile()
    pile = np.copy(regulate)
    print(pile)
    matrice.set_array(pile)
    print("iteration: " +str(i)+" cascade: "+str(cascade))'''
    
if __name__ == "__main__":
   ''' fig, ax = plt.subplots()
    cmap = ListedColormap(['r','w','k','b','y'])
    matrice = ax.matshow(pile,cmap=cmap)
    plt.colorbar(matrice)

    ani = animation.FuncAnimation(fig,update,frames=100,interval = 50)
    plt.show()
    print(pile)'''
   i = 0
   x = list()
   y = list()

   frequency_dict = dict()
   while i != 20000:
        pile = np.copy(drop_sand(pile))
        pile,cascade = regulate_pile(pile)
        print("iteration: " +str(i)+ " number of cascade: "+ str(cascade))
        x.append(i)
        y.append(cascade)
        if cascade not in (frequency_dict.keys()):
            frequency_dict[cascade] = 1
        else:
            frequency_dict[cascade] += 1
        i+=1
   plt.subplot(1,2,1)
   plt.plot(np.array(x),np.array(y),label="Cascade Event")
   plt.scatter(np.array(x),np.array(y),c='y',label="Cascade count")
   plt.title("Search Graph")
   plt.xlabel("Iteration Number")
   plt.ylabel("Casade Number")
   plt.legend()
   print(frequency_dict)

   plt.subplot(1,2,2)
   fr_x = np.sort(np.array(list(frequency_dict.keys())),kind= 'mergesort')
   fr_y = []
   fr_x = fr_x[1:]
   for i in fr_x:
       fr_y.append(frequency_dict[i])
   fr_y = np.array(fr_y)
   fr_y = np.log10(fr_y)
   fr_x = np.log10(fr_x) 


   z= np.polyfit(fr_x,fr_y,deg=1)
   print(z)
   p = np.poly1d(z)
   dp = p(fr_x)

   plt.plot(fr_x,dp,label='fit')
   #plt.plot(fr_x,fr_y,c='g',label="Event Trending")
   #plt.scatter(fr_x,fr_y-dp,label ='detrend')
   plt.scatter(fr_x,fr_y,c ='r',label="occurance count")
   plt.xlabel("Number of Casacade log(10)")
   plt.ylabel("Frequency for given occurance (log10)")
   plt.title(str(z[0])+"*x + "+ str(z[1]))
   plt.legend()
   plt.show()