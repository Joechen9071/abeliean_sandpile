from tkinter import *
from tkinter import font
from tkinter import messagebox
import threading
import tkinter
from turtle import left
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import cv2
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib
import sandpile_redblack
from datetime import datetime
import os
import csv
import sandpile_ver2
from mpl_toolkits.mplot3d import Axes3D #


matplotlib.use('Agg')

stimulation=0
now = datetime.now()
folder_name = now.strftime("%b-%d-%Y.%H.%M.%S")
os.mkdir(folder_name)

fig = plt.figure()
plot1 = fig.add_subplot(2,2,1)
plot2 = fig.add_subplot(2,2,2)
plot3 = fig.add_subplot(2,2,3,projection='3d')
plot4 = fig.add_subplot(2,2,4)
plt.tight_layout(h_pad=2)



def trackable_step(iteration,pile):
    i = 0
    x = list()
    y = list()
    frequency_dict = dict()
    affected_history = list()

    for i in range(iteration):
        sandpile_ver2.drop_sand(pile,i)
        cascade,area_affected = sandpile_ver2.regulate_pile(pile,i)
        print("iteration: " +str(i)+ " number of cascade: "+ str(cascade))
        x.append(i)
        y.append(cascade)

        if cascade not in list(frequency_dict.keys()):
            frequency_dict[cascade] = 1
        else:
            frequency_dict[cascade] += 1
        
        current_iteration.config(text="current iteration: "+ str(i))
        perctange_completion.config(text=str(np.around((i+1)/iteration*100,2)) +"% completed")

        affected_history.append(area_affected)
    
    global stimulation
    stimulation = stimulation+1

    filename = sandpile_ver2.getfilename(folder_name)
    cascade_area = sandpile_ver2.areafilename(folder_name)

    fallout_edges = sandpile_ver2.fallout_edges
    fallout_edges = np.concatenate(fallout_edges)

    with open(folder_name+"/trackable_stimulation_"+str(stimulation)+"_"+filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration","position(y,x)","grain"])
        for i in fallout_edges:
            writer.writerow(i)
    with open(folder_name+"/trackable_stimulation_"+str(stimulation)+"_"+cascade_area, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration","area percentage"])
        for i in range(len(affected_history)):
            writer.writerow([i,int(affected_history[i])])
    

    plot1.plot(np.array(x),np.array(y),label="Cascade Event stimulation: "+str(stimulation))
    plot1.scatter(np.array(x),np.array(y),c='y',label="Cascade count stimulation: "+str(stimulation))
    plot1.title.set_text("Search Graph")
    plot1.set_xlabel("Iteration Number")
    plot1.set_ylabel("Casade Number")
    plot1.legend()

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

    plot2.plot(fr_x,dp,label='fit')
        #plt.plot(fr_x,fr_y,c='g',label="Event Trending")
        #plt.scatter(fr_x,fr_y-dp,label ='detrend')
    plot2.scatter(fr_x,fr_y,c ='r',label="occurance count")
    plot2.set_xlabel("Number of Casacade log(10) stimulation: "+str(stimulation))
    plot2.set_ylabel("Frequency for given occurance stimulation: "+str(stimulation))
    plot2.title.set_text(str(np.around(z[0],4))+"*x + "+ str(np.around(z[1],4)))
    plot2.legend()


    plot4.plot(np.arange(len(affected_history)),np.array(affected_history,dtype=np.int64),c='orange')
    plot4.title.set_text("Area affected")
    plot4.set_xlabel("iteration")
    plot4.set_ylabel("affected area (%)")



    plot3d_pts_x = []
    plot3d_pts_y = []
    plot3d_pts_z = []
    plot3d_pts_c = []
    for y in range(pile.shape[0]):
        for x in range(pile.shape[1]):
            for z in range(pile[y][x].shape[0]):
                plot3d_pts_x.append(x)
                plot3d_pts_y.append(y)
                plot3d_pts_z.append(z)
                plot3d_pts_c.append("r")
    plot3.scatter3D(plot3d_pts_x,plot3d_pts_y,plot3d_pts_z,c=plot3d_pts_c)
    plot3.set_zlim(0,4)
    if stimulation <=1:
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, master)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    else:
        current_stimulation.configure(text="stimulation result: "+str(stimulation))
        canvas.draw()
    master.geometry("1200x1000")

def black_red_queen_step(iteration,pile):
    i = 0
    x = list()
    y = list()
    frequency_dict = dict()
    affected_history = list()

    for i in range(iteration):
        sandpile_redblack.drop_sand(pile,i)
        cascade,area_affected = sandpile_redblack.regulate_pile(pile,i)
        print("iteration: " +str(i)+ " number of cascade: "+ str(cascade))
        x.append(i)
        y.append(cascade)

        if cascade not in list(frequency_dict.keys()):
            frequency_dict[cascade] = 1
        else:
            frequency_dict[cascade] += 1
        
        current_iteration.config(text="current iteration: "+ str(i))
        perctange_completion.config(text=str(np.around((i+1)/iteration*100,2)) +"% completed")

        affected_history.append(area_affected)
    np.savez(now.strftime("%b-%d-%Y.%H.%M.%S"),pile)
    global stimulation
    stimulation = stimulation+1

    filename = sandpile_redblack.getfilename(folder_name)
    cascade_area = sandpile_redblack.areafilename(folder_name)

    fallout_edges = sandpile_redblack.fallout_edges
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
    

    plot1.plot(np.array(x),np.array(y),label="Cascade Event stimulation: "+str(stimulation))
    plot1.scatter(np.array(x),np.array(y),c='y',label="Cascade count stimulation: "+str(stimulation))
    plot1.title.set_text("Search Graph")
    plot1.set_xlabel("Iteration Number")
    plot1.set_ylabel("Casade Number")
    plot1.legend()

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

    plot2.plot(fr_x,dp,label='fit')
        #plt.plot(fr_x,fr_y,c='g',label="Event Trending")
        #plt.scatter(fr_x,fr_y-dp,label ='detrend')
    plot2.scatter(fr_x,fr_y,c ='r',label="occurance count")
    plot2.set_xlabel("Number of Casacade log(10) stimulation: "+str(stimulation))
    plot2.set_ylabel("Frequency for given occurance stimulation: "+str(stimulation))
    plot2.title.set_text(str(np.around(z[0],4))+"*x + "+ str(np.around(z[1],4)))
    plot2.legend()


    plot4.plot(np.arange(len(affected_history)),np.array(affected_history,dtype=np.int64),c='orange')
    plot4.title.set_text("Area affected")
    plot4.set_xlabel("iteration")
    plot4.set_ylabel("affected area (%)")



    plot3d_pts_x = []
    plot3d_pts_y = []
    plot3d_pts_z = []
    plot3d_pts_c = []
    for y in range(pile.shape[0]):
        for x in range(pile.shape[1]):
            for z in range(pile[y][x].shape[0]):
                plot3d_pts_x.append(x)
                plot3d_pts_y.append(y)
                plot3d_pts_z.append(z)
                if pile[y][x][z] == 0:
                    plot3d_pts_c.append('r')
                else:
                    plot3d_pts_c.append('k')
    plot3.scatter3D(plot3d_pts_x,plot3d_pts_y,plot3d_pts_z,c=plot3d_pts_c)
    plot3.set_zlim(0,4)
    if stimulation <=1:
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, master)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    else:
        current_stimulation.configure(text="stimulation result: "+str(stimulation))
        canvas.draw()
    master.geometry("1200x1000")


def confirm_click():
    configured_height = StringVar(root)
    configured_width= StringVar(root)
    configured_iteration = StringVar(root)

    configured_height.set("configured grid height: "+ matrix_height.get())
    configured_width.set("configured grid width: "+ matrix_width.get())
    configured_iteration.set("configured iteration: "+ iteration.get())

    height_config = Label(root,text=configured_height.get())
    width_config = Label(root,text=configured_width.get())
    iteration_config = Label(root,text=configured_iteration.get())
    
    master.geometry("300x400")




def run_stimulation():
    
    try:
        matrix_h = int(matrix_height.get())
    except:
        messagebox.showinfo("Error","grid height must be numerical value")
    try:
        matrix_d = int(matrix_width.get())
    except:
        messagebox.showinfo("Error","grid width must be numerical value")
    
    try:
        iteration_count = int(iteration.get())
    except:
        messagebox.showinfo("Error","grid width must be numerical value")
    
    if selected_model.get() == "Normal":
        pile = np.array(np.zeros((matrix_h,matrix_d),dtype=object))
        pile.fill(np.array([],dtype=int))
        x = threading.Thread(target=trackable_step,args=(iteration_count,pile,))
        x.start()
    elif selected_model.get() == "Black-Red Queen":
        pile = np.empty((matrix_h,matrix_d),dtype=object)
        pile.fill(np.array([],dtype=int))
        x = threading.Thread(target=black_red_queen_step,args=(iteration_count,pile,))
        x.start()


master = Tk()
canvas = FigureCanvasTkAgg(fig, master=master)  # A tk.DrawingArea.

root = master

label1 = Label(root,text="Abliean Sandpile Config",font=font.BOLD,bg="#33ffff")
label2 = Label(root,text="Grid Height:")
label3 = Label(root,text="Grid width:")
label4= Label(root,text="Iteration:")
label5= Label(root,text="Model:")

button1 = Button(root,text="Run stimulation",padx=20,command=run_stimulation)
button2 = Button(root,text="Confirm",padx=5,command=confirm_click)

matrix_height = Entry(root,width=20,bg="white",bd=5,justify="left")
matrix_width = Entry(root,width=20,bg="white",bd=5,justify="left")
iteration = Entry(root,width=20,bg="white",bd=5,justify="left")

selected_model= StringVar(root)
selected_model.set("Normal")
model_type= OptionMenu(root, selected_model, "Normal", "Black-Red Queen")


#title
label1.pack()

#model selection
label5.pack()
model_type.pack()

#grid height ui
label2.pack()
matrix_height.pack()

#grid width ui
label3.pack()
matrix_width.pack()

#iteration ui
label4.pack()
iteration.pack()

#button ui
button2.pack()
button1.pack()

current_iteration = Label(root,text="current iteration: "+ str("None"))
current_iteration.pack()
perctange_completion = Label(root,text=str("None") +"% completed")
perctange_completion.pack()

text_var = StringVar(root)
text_var.set("stimulation result: "+str(stimulation+1))
current_stimulation = Label(root,text=text_var.get())
current_stimulation.pack()


master.geometry('300x350')
master.title("Abliean Sandpile stimulator")
master.configure(background="#ffff66")
master.mainloop()