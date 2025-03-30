import tkinter as tk
import tkinter.messagebox as tkmsg
from pathlib import Path
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Path to asset files for this GUI window.
ASSETS_PATH = Path(__file__).resolve().parent / "assets"

# Function that takes input and passes to the model
def btn_clicked():
    city = city_entry.get()
    pH = pH_entry.get()
    temperature = 33
    humidity = 75
    rainfall = 196
    inputs = pd.read_csv(Path(__file__).resolve().parent/"Crop_recommendation.csv")


    le_label = LabelEncoder()

    inputs['label_num'] = le_label.fit_transform(inputs["label"])

    # To check the labels
    # print(inputs['label'].value_counts())
    
    inputs_n = inputs.drop(["N","P","K","label","label_num"],axis = "columns")
    target = inputs["label_num"]

    model = tree.DecisionTreeClassifier()

    model.fit(inputs_n.values,target)

    # # To predict our values
    if(city != '' and pH !=''):
        ans = model.predict([[temperature,humidity,float(pH),rainfall]])
        ans = le_label.inverse_transform(ans)[-1]
        tkmsg.showinfo("Result!",f"The predicted crop in {city.capitalize()} is {ans.capitalize()}")
    else:
        tkmsg.showerror("Error!","You hadn't entered all the details")
window = tk.Tk()
window.title("Crop Predictor v1.0")
window.geometry("862x519")
icon_img = tk.PhotoImage(file=ASSETS_PATH / "sprout.png")
window.iconphoto(False,icon_img)
window.configure(bg="#2D5A27")
canvas = tk.Canvas(
    window, bg="#2D5A27", height=519, width=862,
    bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)
canvas.create_rectangle(431, 0, 431 + 431, 0 + 519, fill="#FCFCFC", outline="")
canvas.create_rectangle(40, 160, 40 + 60, 160 + 5, fill="#FCFCFC", outline="")

text_box_bg = tk.PhotoImage(file=ASSETS_PATH / "TextBox_Bg.png")
city_entry_img = canvas.create_image(650.5, 167.5, image=text_box_bg)
pH_entry_img = canvas.create_image(650.5, 248.5, image=text_box_bg)

city_entry = tk.Entry(bd=0, bg="#F6F7F9",fg="#000716",  highlightthickness=0)
city_entry.place(x=490.0, y=137+25, width=321.0, height=35)
city_entry.focus()

pH_entry = tk.Entry(bd=0, bg="#F6F7F9", fg="#000716",  highlightthickness=0)
pH_entry.place(x=490.0, y=218+25, width=321.0, height=35)


canvas.create_text(
    490.0, 150.0, text="Name of the City", fill="#3F7E36",
    font=("Arial-BoldMT", int(13.0)), anchor="w")
canvas.create_text(
    490.0, 229.5, text="pH value of the soil", fill="#3F7E36",
    font=("Arial-BoldMT", int(13.0)), anchor="w")
canvas.create_text(
    646.5, 428.5, text="Generate",
    fill="#FFFFFF", font=("Arial-BoldMT", int(13.0)))
canvas.create_text(
    573.5, 88.0, text="Enter the details.",
    fill="#3F7E36", font=("Arial-BoldMT", int(22.0)))

title = tk.Label(
    text="Crop Predictor v1.0", bg="#2D5A27",
    fg="white", font=("Arial-BoldMT", int(20.0)))
title.place(x=27.0, y=120.0)

info_text = tk.Label(
    text="Crop Predictor predicts the type of\n"
    "crop to plant in an area using\n"
    "real time weather data from\n"
    "Weather API.\n\n"

    "Information later processed using\n"
    "Decision Tree Algorithm.",
    bg="#2D5A27", fg="white", justify="left",
    font=("Georgia", int(16.0)))

info_text.place(x=27.0, y=200.0)


generate_btn_img = tk.PhotoImage(file=ASSETS_PATH / "generate.png")
generate_btn = tk.Button(
    image=generate_btn_img, borderwidth=0, highlightthickness=0,
    command=btn_clicked, relief="flat")
generate_btn.place(x=557, y=390, width=180, height=55)

window.resizable(False, False)
window.mainloop()