import tkinter as tk

import os
from PIL import Image, ImageTk

top = tk.Tk()
top.geometry("740x525")
top.title("Plant disease and yield recognition")

count =0
image=0
def change_frame(label1):
    def show_frame():
        global count
        global image
        if (count!=20):
            my_img = Image.open("/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/out/frame"+str(image)+".jpg")
            my_img = my_img.resize((350, 350), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(my_img)
            label1.configure(image=photo)
            label1.image=photo
            label1.place(x=450, y=250, anchor="center")
            count+=1
            image+=1
            label1.after(5000,show_frame)
        else:
            label1.destroy()
    show_frame()

def showframeresults():
    change_frame(label1)

label1 = tk.Label(top)
imgopenbtn = tk.Button(top,text="open", command=showframeresults, height=2, width=5)
imgopenbtn.place(x=0,y=0)
print(len(os.listdir("/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/out/frameout/resizedframes/")))
top.mainloop()

