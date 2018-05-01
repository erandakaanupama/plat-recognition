from tkinter import filedialog, ttk
import tkinter as tk
import PIL
from PIL import Image, ImageTk
from plant_rec.plantrec import *
from Plant_deficieny_identification_module.DeficiencyIdentificationTest2 import *
# from plant_yield2.anotherPythonApplication import get_detections


# added champ /home/chamod/disease_yieldrec
local_dir = "/home/erandaka/PycharmProjects/" # if project path is /home/erandaka/PycharmProjects/disease_yieldrec local_dir should be /home/erandaka/PycharmProjects/
#local_dir = "/home/chamod/" # if project path is /home/erandaka/PycharmProjects/disease_yieldrec local_dir should be /home/erandaka/PycharmProjects/

top = tk.Tk()
top.geometry("740x550")
top.title("Plant disease and yield recognition")

# directoryopen = "/home/erandaka/My Studies/Sem8/Prj/data_set/Plants_Original DS"
directoryopen = "/home/erandaka/PycharmProjects/tst-prjc/plant-disease-rec/disease_yieldrec/plant_rec/test_imges"

# handle open image/ video file
def openimgefile():
    global directoryopen
    top.filename = filedialog.askopenfilename(initialdir=directoryopen, title="Select file",
                                              filetypes=(
                                                  ("image files", "*.jpg"), ("video files", "*.mp4"),

                                                  ("all files", "*.*")))
    global imgopenbtn
    global file_name
    global my_img
    global photo
    # handle image file open
    if (".jpg" in top.filename):

        file_name = top.filename
        directoryopen = file_name.split(os.path.basename(file_name))[0]
        my_img = Image.open(top.filename)
        my_img = my_img.resize((350, 350), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(my_img)
        imgopenbtn.config(image=photo)

    elif (".mp4" in top.filename):
        file_name = top.filename

        directoryopen = file_name.split(os.path.basename(file_name))[0]
        vidcap = cv2.VideoCapture(top.filename)
        success, img = vidcap.read()
        (height, width) = img.shape[:2]

        # resizing
        frame_prv_path = local_dir + "disease_yieldrec/plant_rec/out/frame_prev/"
        resize_width = 912
        r = resize_width / float(width)
        dim = (resize_width, int(height * r))
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(frame_prv_path + "frame%d.jpg" % 0, resized_img)
        my_img = Image.open(frame_prv_path + "/frame0.jpg")
        my_img = my_img.resize((350, 350), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(my_img)
        imgopenbtn.config(image=photo)


# open image for predict
my_img = Image.open(local_dir + "disease_yieldrec/ui-images/add-file.png")

my_img = my_img.resize((150, 150), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(my_img)
imgopenbtn = tk.Button(top, image=photo, command=openimgefile, height=350, width=350)
imgopenbtn.place(x=5, y=5)


# predictions for species, disease, yield
# predictions for disease

def diseaserec():

    if (".jpg" in file_name):
        global canvas
        canvas.delete("all")
        imgeresultlbl.delete('1.0', tk.END)

        image_file = os.path.basename(file_name)
        file_path = file_name.split(image_file)[0]

        base_width = 912
        save_path = local_dir + "disease_yieldrec/plant_rec/out/resizedimg_predict/"

        img = Image.open(file_name)
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), PIL.Image.ANTIALIAS)
        img.save(save_path + image_file)

        outscores, _, classnames, prd_image,_ = predict(sess, save_path, image_file)
        prd_image.save(os.path.join("out", image_file), quality=90)

        # show image category
        trainedclasses = ["bellpepper", "tomato", "beans", "lettuce", "gherkins"]

        maxscore = (max(outscores))
        maxscoreindex = 0
        for i in range(0, len(outscores)):
            if (outscores[i] == maxscore):
                maxscoreindex = i

        classindex = classnames[
            maxscoreindex]  # class index return, 0, 1, 2, 3, 4 for "bellpepper", "tomato", "beans", "lettuce", "gherkins"

        result=deficiency_predict(save_path, image_file,classindex)

        #  dict_deficincies = {1.0: "Gherkin-Mg", 2.0: "Tomatoes-Ca", 3.0: "Beans-Mg", 4.0: "BellPepper-N"}

        # preview result
        previmg = Image.open(
            local_dir + "disease_yieldrec/Plant_deficieny_identification_module/out/" + image_file)
        previmg = previmg.resize((350, 350), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(previmg)
        canvas.image = photo  # <--- keep reference of your image
        canvas.create_image(0, 0, anchor='nw', image=photo)

        # do the stuff for disease recognition with help of identified plants and update plantdisese dictionary (classindex)
        plantdisese = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0
        }

        # showing progress bars

        x_pos_pb = 458
        y_pos = 380
        x_pos_lb = 360

        global pb_bp
        global bp_lbl
        global pb_tomt
        global tomt_lbl
        global pb_bns
        global bns_lbl
        global pb_let
        global let_lbl
        global pb_ghr
        global ghr_lbl

        pb_bp.destroy()
        bp_lbl.destroy()
        pb_tomt.destroy()
        tomt_lbl.destroy()
        pb_bns.destroy()
        bns_lbl.destroy()
        pb_let.destroy()
        let_lbl.destroy()
        pb_ghr.destroy()
        ghr_lbl.destroy()

        global gherkin_mg_pb
        global gherkin_mg_lbl
        global tomatoes_ca_pb
        global tomatoes_ca_lbl
        global beans_mg_pb
        global beans_mg_lbl
        global bellPepper_n_pb
        global bellPepper_n_lbl

        gherkin_mg_pb.destroy()
        gherkin_mg_lbl.destroy()
        tomatoes_ca_pb.destroy()
        tomatoes_ca_lbl.destroy()
        beans_mg_pb.destroy()
        beans_mg_lbl.destroy()
        bellPepper_n_pb.destroy()
        bellPepper_n_lbl.destroy()

        if (1 in result and result[1] > 0.1):
            gherkin_mg_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            gherkin_mg_lbl = tk.Label(top, text="Gherkin_Mg")
            gherkin_mg_pb.pack()
            gherkin_mg_pb["maximum"] = 100
            gherkin_mg_pb["value"] = int(result[1] * 100)
            gherkin_mg_pb.place(x=x_pos_pb, y=y_pos)
            gherkin_mg_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (2 in result and result[2] > 0.1):
            tomatoes_ca_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            tomatoes_ca_pb.pack()
            tomatoes_ca_pb["maximum"] = 100
            tomatoes_ca_pb["value"] = int(result[2] * 100)
            tomatoes_ca_pb.place(x=x_pos_pb, y=y_pos)
            tomatoes_ca_lbl = tk.Label(top, text="Tomatoes_Ca")
            tomatoes_ca_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (3 in result and result[3] > 0.1):
            beans_mg_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            beans_mg_pb.pack()
            beans_mg_pb["maximum"] = 100
            beans_mg_pb["value"] = int(result[3] * 100)
            beans_mg_pb.place(x=x_pos_pb, y=y_pos)
            beans_mg_lbl = tk.Label(top, text="Beans_Mg")
            beans_mg_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (4 in result and result[4] > 0.1):
            bellPepper_n_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            bellPepper_n_pb.pack()
            bellPepper_n_pb["maximum"] = 100
            bellPepper_n_pb["value"] = int(result[4] * 100)
            bellPepper_n_pb.place(x=x_pos_pb, y=y_pos)
            bellPepper_n_lbl = tk.Label(top, text="BellPepper_N")
            bellPepper_n_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22


    elif (".mp4" in file_name):
        # for videos

        global vidcap
        vidcap = cv2.VideoCapture(file_name)
        change_frame_fordiseasepred(frameprevlbl)

def change_frame_fordiseasepred(label1):

    #label1.destroy()

    pb_bp.destroy()
    bp_lbl.destroy()
    pb_tomt.destroy()
    tomt_lbl.destroy()
    pb_bns.destroy()
    bns_lbl.destroy()
    pb_let.destroy()
    let_lbl.destroy()
    pb_ghr.destroy()
    ghr_lbl.destroy()
    imgeresultlbl.delete('1.0', tk.END)


    gherkin_mg_pb.destroy()
    gherkin_mg_lbl.destroy()
    tomatoes_ca_pb.destroy()
    tomatoes_ca_lbl.destroy()
    beans_mg_pb.destroy()
    beans_mg_lbl.destroy()
    bellPepper_n_pb.destroy()
    bellPepper_n_lbl.destroy()

    def show_frame():

        global count
        global vidcap
        success, img = vidcap.read()
        if (success == True):
            if (count % 10 == 0):
                (height, width) = img.shape[:2]

                # resizing
                resize_width = 912
                r = resize_width / float(width)
                dim = (resize_width, int(height * r))
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(frame_path + "frame%d.jpg" % count, resized_img)
                outscores, _, classnames, prd_image,_ = predict(sess,
                                                              frame_path,
                                                              "frame%d.jpg" % count)


                #prd_image.save(os.path.join("out", "frame%d.jpg"%count), quality=90)

                # show image category
                trainedclasses = ["bellpepper", "tomato", "beans", "lettuce", "gherkins"]

                maxscore = (max(outscores))
                maxscoreindex = 0
                for i in range(0, len(outscores)):
                    if (outscores[i] == maxscore):
                        maxscoreindex = i

                classindex = classnames[
                    maxscoreindex]  # class index return, 0, 1, 2, 3, 4 for "bellpepper", "tomato", "beans", "lettuce", "gherkins"

                save_path = local_dir+"disease_yieldrec/plant_rec/out/frameout/"
                result = deficiency_predict(save_path, "frame%d.jpg"%count, classindex)

                #prd_image.save(os.path.join("out", "frame%d.jpg" % count), quality=90)

                # show image previewyhl1
                my_img = Image.open(
                    local_dir + "disease_yieldrec/Plant_deficieny_identification_module/out/frame" + str(count) + ".jpg")
                my_img = my_img.resize((350, 350), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(my_img)
                label1.configure(image=photo)
                label1.image = photo
                label1.place(x=550, y=185, anchor="center")
                count += 1

                # show pg bars

                x_pos_pb = 458
                y_pos = 380
                x_pos_lb = 360

                pb_bp.destroy()
                bp_lbl.destroy()
                pb_tomt.destroy()
                tomt_lbl.destroy()
                pb_bns.destroy()
                bns_lbl.destroy()
                pb_let.destroy()
                let_lbl.destroy()
                pb_ghr.destroy()
                ghr_lbl.destroy()

                global gherkin_mg_pb
                global gherkin_mg_lbl
                global tomatoes_ca_pb
                global tomatoes_ca_lbl
                global beans_mg_pb
                global beans_mg_lbl
                global bellPepper_n_pb
                global bellPepper_n_lbl

                gherkin_mg_pb.destroy()
                gherkin_mg_lbl.destroy()
                tomatoes_ca_pb.destroy()
                tomatoes_ca_lbl.destroy()
                beans_mg_pb.destroy()
                beans_mg_lbl.destroy()
                bellPepper_n_pb.destroy()
                bellPepper_n_lbl.destroy()
                imgeresultlbl.delete('1.0', tk.END)

                if (1 in result and result[1] > 0.01):
                    gherkin_mg_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    gherkin_mg_lbl = tk.Label(top, text="Gherkin_Mg")
                    gherkin_mg_pb.pack()
                    gherkin_mg_pb["maximum"] = 100
                    gherkin_mg_pb["value"] = int(result[1] * 100)
                    gherkin_mg_pb.place(x=x_pos_pb, y=y_pos)
                    gherkin_mg_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (2 in result and result[2] > 0.01):
                    tomatoes_ca_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    tomatoes_ca_pb.pack()
                    tomatoes_ca_pb["maximum"] = 100
                    tomatoes_ca_pb["value"] = int(result[2] * 100)
                    tomatoes_ca_pb.place(x=x_pos_pb, y=y_pos)
                    tomatoes_ca_lbl = tk.Label(top, text="Tomatoes_Ca")
                    tomatoes_ca_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (3 in result and result[3] > 0.01):
                    beans_mg_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    beans_mg_pb.pack()
                    beans_mg_pb["maximum"] = 100
                    beans_mg_pb["value"] = int(result[3] * 100)
                    beans_mg_pb.place(x=x_pos_pb, y=y_pos)
                    beans_mg_lbl = tk.Label(top, text="Beans_Mg")
                    beans_mg_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (4 in result and result[4] > 0.01):
                    bellPepper_n_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    bellPepper_n_pb.pack()
                    bellPepper_n_pb["maximum"] = 100
                    bellPepper_n_pb["value"] = int(result[4] * 100)
                    bellPepper_n_pb.place(x=x_pos_pb, y=y_pos)
                    bellPepper_n_lbl = tk.Label(top, text="BellPepper_N")
                    bellPepper_n_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22
                label1.after(500, show_frame)

            else:
                count += 1
                label1.after(500, show_frame)
        else:

            count = 0

    show_frame()


# # # K deficiency progress bar
# k_defpb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
# k_defpb.pack()
# k_defpb["maximum"] = 100
# k_defpb["value"] = int(list(resultNew.values())[2])
# n_defpb=tk.Label(top, text=list(resultNew.values())[2])
# k_defpb.place(x=450, y=424)
#
# maitaatkpb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
# maitaatkpb.pack()
# maitaatkpb["maximum"] = 100
# maitaatkpb["value"] = int(list(resultNew.values())[3])
# n_defpb=tk.Label(top, text=list(resultNew.values())[3])
# maitaatkpb.place(x=450, y=446)




# analyze disease
predictbtn = tk.Button(top, text="Disease Check", command=diseaserec)
predictbtn.place(x=105, y=410)

# check plant rec results
file_name = ""


def plantrec():

    imgeresultlbl.delete('1.0', tk.END)
    if (".jpg" in file_name):
        global canvas
        canvas.delete("all")

        image_file = os.path.basename(file_name)
        file_path = file_name.split(image_file)[0]

        base_width = 912
        save_path = local_dir + "disease_yieldrec/plant_rec/out/resizedimg_predict/"

        img = Image.open(file_name)
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), PIL.Image.ANTIALIAS)
        img.save(save_path + image_file)

        outscores, _, classnames, prd_image, box_coordinates = predict(sess, save_path, image_file)
        prd_image.save(os.path.join("out", image_file), quality=90)

        # update txt box entry

        left_cor = box_coordinates["0"]
        top_cor = box_coordinates["1"]
        rgt_cor = box_coordinates["2"]
        btm_cor = box_coordinates["3"]

        for i in range(0,len(left_cor)):
            imgeresultlbl.insert(tk.INSERT,"("+str(left_cor[i])+","+str(top_cor[i])+")"+"("+str(rgt_cor[i])+","+str(btm_cor[i])+")\n")
            imgeresultlbl.insert(tk.INSERT,"")

        # for i in range(0,len(left)):
        #     print("from predictions")
        #     print(left[i])

        # show image category
        trainedclasses = ["bellpepper", "tomato", "beans", "lettuce", "gherkins"]

        plantreslts = {"0": 0,
                       "1": 0,
                       "2": 0,
                       "3": 0,
                       "4": 0}

        # update resultsdictionary
        for j in range(0, len(classnames)):
            classcategory = classnames[j]
            if (plantreslts[str(classcategory)] < outscores[j]):
                plantreslts[str(classcategory)] = outscores[j]

        """maxscore = (max(outscores))
        maxscoreindex=0
        for i in range(0,len(outscores)):
            if(outscores[i]==maxscore):
                maxscoreindex=i

        classindex=classnames[maxscoreindex]
        global imgeresultlbl
        imgeresultlbl.delete('1.0', tk.END)"""
        """imgeresultlbl.insert(tk.INSERT, classnames[maxscoreindex])
        imgeresultlbl.insert(tk.INSERT, outscores)
        imgeresultlbl.insert(tk.INSERT, classnames)
        imgeresultlbl.insert(tk.INSERT, trainedclasses[classindex])
        imgeresultlbl.place(x=490, y=360)"""

        # show progress bars
        x_pos_pb = 450
        y_pos = 380
        x_pos_lb = 380

        global pb_bp
        global bp_lbl
        global pb_tomt
        global tomt_lbl
        global pb_bns
        global bns_lbl
        global pb_let
        global let_lbl
        global pb_ghr
        global ghr_lbl

        pb_bp.destroy()
        bp_lbl.destroy()
        pb_tomt.destroy()
        tomt_lbl.destroy()
        pb_bns.destroy()
        bns_lbl.destroy()
        pb_let.destroy()
        let_lbl.destroy()
        pb_ghr.destroy()
        ghr_lbl.destroy()



        gherkin_mg_pb.destroy()
        gherkin_mg_lbl.destroy()
        tomatoes_ca_pb.destroy()
        tomatoes_ca_lbl.destroy()
        beans_mg_pb.destroy()
        beans_mg_lbl.destroy()
        bellPepper_n_pb.destroy()
        bellPepper_n_lbl.destroy()

        if (plantreslts["0"] > 0):
            pb_bp = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            bp_lbl = tk.Label(top, text=trainedclasses[0])
            pb_bp.pack()
            pb_bp["maximum"] = 100
            pb_bp["value"] = int(plantreslts["0"] * 100)
            pb_bp.place(x=x_pos_pb, y=y_pos)
            bp_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (plantreslts["1"] > 0):
            pb_tomt = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            pb_tomt.pack()
            pb_tomt["maximum"] = 100
            pb_tomt["value"] = int(plantreslts["1"] * 100)
            pb_tomt.place(x=x_pos_pb, y=y_pos)
            tomt_lbl = tk.Label(top, text=trainedclasses[1])
            tomt_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (plantreslts["2"] > 0):
            pb_bns = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            pb_bns.pack()
            pb_bns["maximum"] = 100
            pb_bns["value"] = int(plantreslts["2"] * 100)
            pb_bns.place(x=x_pos_pb, y=y_pos)
            bns_lbl = tk.Label(top, text=trainedclasses[2])
            bns_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (plantreslts["3"] > 0):
            pb_let = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            pb_let.pack()
            pb_let["maximum"] = 100
            pb_let["value"] = int(plantreslts["3"] * 100)
            pb_let.place(x=x_pos_pb, y=y_pos)
            let_lbl = tk.Label(top, text=trainedclasses[3])
            let_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        if (plantreslts["4"] > 0):
            pb_ghr = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
            pb_ghr.pack()
            pb_ghr["maximum"] = 100
            pb_ghr["value"] = int(plantreslts["4"] * 100)
            pb_ghr.place(x=x_pos_pb, y=y_pos)
            ghr_lbl = tk.Label(top, text=trainedclasses[4])
            ghr_lbl.place(x=x_pos_lb, y=y_pos)
            y_pos += 22

        # update text field

        # preview result
        previmg = Image.open(local_dir + "disease_yieldrec/plant_rec/out/" + image_file)
        previmg = previmg.resize((350, 350), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(previmg)
        canvas.image = photo  # <--- keep reference of your image
        canvas.create_image(0, 0, anchor='nw', image=photo)

    elif (".mp4" in file_name):
        global vidcap
        vidcap = cv2.VideoCapture(file_name)
        change_frame(frameprevlbl)


# add canvas for image result preview
canvas = tk.Canvas(height=350, width=350)
canvas.pack()
canvas.place(x=380, y=8)

# pg bars and labels for plant rec

pb_bp = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
bp_lbl = tk.Label(top, text="bellpepper")
pb_tomt = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
tomt_lbl = tk.Label(top, text="tomato")
pb_bns = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
bns_lbl = tk.Label(top, text="beans")
pb_let = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
let_lbl = tk.Label(top, text="lettuce")
pb_ghr = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
ghr_lbl = tk.Label(top, text="gherkins")

# progress bars and labels for disese rec
gherkin_mg_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
gherkin_mg_lbl = tk.Label(top, text="Gherkin_Mg")
tomatoes_ca_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
tomatoes_ca_lbl = tk.Label(top, text="Tomatoes_Ca")
beans_mg_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
beans_mg_lbl = tk.Label(top, text="Beans_Mg")
bellPepper_n_pb = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
bellPepper_n_lbl = tk.Label(top, text="BellPepper_N")


# text field entry
imgeresultlbl = tk.Text(top, height=2, width=35)
imgeresultlbl.insert(tk.INSERT, "add an image for preview image")
imgeresultlbl.pack()
imgeresultlbl.place(x=425, y=450)
"""previmg = Image.open("/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/out/frameout/resizedframes/" +"frame0"+".jpg")
previmg = previmg.resize((350, 350), Image.ANTIALIAS)
photoprv = ImageTk.PhotoImage(previmg)
canvasimg=canvas.create_image(0, 0, anchor='nw', image=photoprv)"""

# frames preview
count = 0
image = 0
success = False
frame_path = local_dir + "disease_yieldrec/plant_rec/out/frameout/"
vidcap = cv2.VideoCapture(file_name)


def change_frame(label1):

    canvas.delete("all")

    pb_bp.destroy()
    bp_lbl.destroy()
    pb_tomt.destroy()
    tomt_lbl.destroy()
    pb_bns.destroy()
    bns_lbl.destroy()
    pb_let.destroy()
    let_lbl.destroy()
    pb_ghr.destroy()
    ghr_lbl.destroy()
    imgeresultlbl.delete('1.0', tk.END)


    gherkin_mg_pb.destroy()
    gherkin_mg_lbl.destroy()
    tomatoes_ca_pb.destroy()
    tomatoes_ca_lbl.destroy()
    beans_mg_pb.destroy()
    beans_mg_lbl.destroy()
    bellPepper_n_pb.destroy()
    bellPepper_n_lbl.destroy()

    def show_frame():

        global count
        global vidcap
        success, img = vidcap.read()
        if (success == True):
            if (count % 10 == 0):
                (height, width) = img.shape[:2]

                # resizing
                resize_width = 912
                r = resize_width / float(width)
                dim = (resize_width, int(height * r))
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(frame_path + "frame%d.jpg" % count, resized_img)
                outscores, _, classnames, prd_image = predict(sess,
                                                              frame_path,
                                                              "frame%d.jpg" % count)

                prd_image.save(os.path.join("out", "frame%d.jpg" % count), quality=90)

                # show image previewyhl1
                my_img = Image.open(
                    local_dir + "disease_yieldrec/plant_rec/out/frame" + str(count) + ".jpg")
                my_img = my_img.resize((350, 350), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(my_img)
                label1.configure(image=photo)
                label1.image = photo
                label1.place(x=550, y=185, anchor="center")
                count += 1

                # show pg bars
                trainedclasses = ["bellpepper", "tomato", "beans", "lettuce", "gherkins"]

                plantreslts = {"0": 0,
                               "1": 0,
                               "2": 0,
                               "3": 0,
                               "4": 0}

                # update resultsdictionary
                for j in range(0, len(classnames)):
                    classcategory = classnames[j]
                    if (plantreslts[str(classcategory)] < outscores[j]):
                        plantreslts[str(classcategory)] = outscores[j]

                x_pos_pb = 450
                y_pos = 380
                x_pos_lb = 380

                global pb_bp
                global bp_lbl
                global pb_tomt
                global tomt_lbl
                global pb_bns
                global bns_lbl
                global pb_let
                global let_lbl
                global pb_ghr
                global ghr_lbl

                pb_bp.destroy()
                bp_lbl.destroy()
                pb_tomt.destroy()
                tomt_lbl.destroy()
                pb_bns.destroy()
                bns_lbl.destroy()
                pb_let.destroy()
                let_lbl.destroy()
                pb_ghr.destroy()
                ghr_lbl.destroy()

                imgeresultlbl.delete('1.0', tk.END)

                gherkin_mg_pb.destroy()
                gherkin_mg_lbl.destroy()
                tomatoes_ca_pb.destroy()
                tomatoes_ca_lbl.destroy()
                beans_mg_pb.destroy()
                beans_mg_lbl.destroy()
                bellPepper_n_pb.destroy()
                bellPepper_n_lbl.destroy()

                if (plantreslts["0"] > 0):
                    pb_bp = tk.ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    bp_lbl = tk.Label(top, text=trainedclasses[0])
                    pb_bp.pack()
                    pb_bp["maximum"] = 100
                    pb_bp["value"] = int(plantreslts["0"] * 100)
                    pb_bp.place(x=x_pos_pb, y=y_pos)
                    bp_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (plantreslts["1"] > 0):
                    pb_tomt = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    pb_tomt.pack()
                    pb_tomt["maximum"] = 100
                    pb_tomt["value"] = int(plantreslts["1"] * 100)
                    pb_tomt.place(x=x_pos_pb, y=y_pos)
                    tomt_lbl = tk.Label(top, text=trainedclasses[1])
                    tomt_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (plantreslts["2"] > 0):
                    pb_bns = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    pb_bns.pack()
                    pb_bns["maximum"] = 100
                    pb_bns["value"] = int(plantreslts["2"] * 100)
                    pb_bns.place(x=x_pos_pb, y=y_pos)
                    bns_lbl = tk.Label(top, text=trainedclasses[2])
                    bns_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (plantreslts["3"] > 0):
                    pb_let = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    pb_let.pack()
                    pb_let["maximum"] = 100
                    pb_let["value"] = int(plantreslts["3"] * 100)
                    pb_let.place(x=x_pos_pb, y=y_pos)
                    let_lbl = tk.Label(top, text=trainedclasses[3])
                    let_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22

                if (plantreslts["4"] > 0):
                    pb_ghr = ttk.Progressbar(top, orient="horizontal", length=200, mode="determinate")
                    pb_ghr.pack()
                    pb_ghr["maximum"] = 100
                    pb_ghr["value"] = int(plantreslts["4"] * 100)
                    pb_ghr.place(x=x_pos_pb, y=y_pos)
                    ghr_lbl = tk.Label(top, text=trainedclasses[4])
                    ghr_lbl.place(x=x_pos_lb, y=y_pos)
                    y_pos += 22
                label1.after(100, show_frame)

            else:
                count += 1
                label1.after(100, show_frame)
        else:

            count = 0

    show_frame()


frameprevlbl = tk.Label(top)

# plant recognition button
plntrecbtn = tk.Button(top, text="Plant species check", command=plantrec)
plntrecbtn.place(x=90, y=380)


# yield calculation
def yieldcalc():
    global yieldtxtbx
    yieldtxtbx.delete('1.0', tk.END)
    yieldtxtbx.insert(tk.INSERT, "calculation")
    if (".jpg" in file_name):
        print("please select a video")

    elif (".mp4" in file_name):
        # for videos

        global vidcap
        vidcap = cv2.VideoCapture(file_name)
        change_frame_for_yield_calc(frameprevlbl)

def change_frame_for_yield_calc(label1):


    def show_frame():

        global count
        global vidcap
        success, img = vidcap.read()
        if (success == True):
            if (count % 10 == 0):
                (height, width) = img.shape[:2]

                # resizing
                resize_width = 912
                r = resize_width / float(width)
                dim = (resize_width, int(height * r))
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(frame_path + "frame%d.jpg" % count, resized_img)
                outscores, _, classnames, prd_image = predict(sess,
                                                              frame_path,
                                                              "frame%d.jpg" % count)


                save_path = local_dir+"disease_yieldrec/plant_rec/out/frameout/"
                result = deficiency_predict(save_path, "frame%d.jpg"%count, 0) # result from erandaka's part

                #prd_image.save(os.path.join("out", "frame%d.jpg" % count), quality=90)

                # show image previewyhl1
                my_img = Image.open(
                    local_dir + "disease_yieldrec/plant_rec/out/frame" + str(count) + ".jpg")
                my_img = my_img.resize((350, 350), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(my_img)
                label1.configure(image=photo)
                label1.image = photo
                label1.place(x=550, y=185, anchor="center")
                count += 1
                my_image=cv2.imread(local_dir + "disease_yieldrec/plant_rec/out/frame" + str(count) + ".jpg")
                #result_for_the_frame=get_detections(my_image)

                # show pg bars

                x_pos_pb = 458
                y_pos = 380
                x_pos_lb = 360



                pb_bp.destroy()
                bp_lbl.destroy()
                pb_tomt.destroy()
                tomt_lbl.destroy()
                pb_bns.destroy()
                bns_lbl.destroy()
                pb_let.destroy()
                let_lbl.destroy()
                pb_ghr.destroy()
                ghr_lbl.destroy()


                label1.after(500, show_frame)

            else:
                count += 1
                label1.after(500, show_frame)
        else:

            count = 0

    show_frame() #recursive call

# yield calculation button
yieldcalbtn = tk.Button(top, text="Yield calculation:", command=yieldcalc)
yieldcalbtn.place(x=99, y=440)

# display textbox
""""yieldtxtbx.insert(tk.INSERT, "add an image for yield calculation")
yieldtxtbx.pack()
yieldtxtbx.place(x=425, y=500)"""

top.mainloop()
