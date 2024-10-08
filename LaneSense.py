from tkinter import *
from tkinter.messagebox import showinfo
import tkinter as tk
# importing libraries
import cv2
import seaborn as sns
import math
import time
import numpy as np
import datetime
from matplotlib import pyplot as plt
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from PIL import ImageTk, Image


def main():
    # OBJECT DETECTION

    # PLOT SETTINGS CONFIG
    sns.set_style("darkgrid")

    e_ws = Tk()
    e_ws.title('Lane Asssesment')
    e_ws.geometry('1366x768')
    bg = ImageTk.PhotoImage(Image.open("assets\images\home.png"))
    canvas = Canvas(e_ws, width=1366, height=768)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg, anchor="nw")

    # Back BUTTON
    def start():
        e_ws.destroy()

    def get_coordinates(event, x, y, flags, param):
        global circle_center
        if event == cv2.EVENT_LBUTTONDOWN:  # Check if left mouse button is clicked
            circle_center = (x, y)
            cv2.setMouseCallback('Imageconv', lambda *args: None)  # Remove mouse callback
            cv2.destroyAllWindows()

    def select_file():
        global filename
        global name
        global phnno
        global address
        filetypes = (
            ('All files', '*.*'),
            ('text files', '*.txt')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            filetypes=filetypes)
        name = entry1.get()
        phnno = entry2.get()
        address = entry3.get()

        start()

        e_ws = Tk()
        e_ws.title('Lane Asssesment')
        e_ws.geometry('1366x768')
        canvas = Canvas(e_ws, width=1366, height=768)
        canvas.pack(fill="both", expand=True)

        cap = cv2.VideoCapture(filename)

        ##CALIB & INITIAL

        circle_radius = 50
        i = 0
        wrong = 0
        correct = 0
        deviation = []

        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])

        calibimg = cap.read()[1]

        sf = 700
        resize_val = int((calibimg.shape[1] / calibimg.shape[0]) * sf)

        calibimg = cv2.resize(calibimg, (resize_val, sf))

        cv2.imshow('Imageconv', calibimg)
        cv2.setMouseCallback('Imageconv', get_coordinates)
        cv2.waitKey()

        # CAM DISPLAY
        embed = Label(e_ws, width=int(resize_val / 2), height=int(sf / 2))
        embed.place(x=1, y=1)

        embed2 = Label(e_ws, width=640, height=480)
        embed2.place(x=800, y=1)

        ####GRAPH-------------------
        cutoffDev = 40

        ###----------------------

        while True:
            # GRPAH

            x_values = range(len(deviation))
            sns.lineplot(x=x_values, y=deviation)  # Plotting the data
            # Shading the region based on the condition
            plt.fill_between(x_values, deviation, where=[val > cutoffDev for val in deviation],
                             color='red', alpha=0.3)  # Shading region where y > 10
            plt.fill_between(x_values, deviation, where=[val <= cutoffDev for val in deviation],
                             color='skyblue', alpha=0.3)  # Shading region where y <= 10

            plt.title(f'Graph of Deviation SCORE:- {np.median(deviation):.2f}')
            plt.ylabel('Deviation Value')
            plt.savefig('assets/temp/plotimage.png')

            graphimg = cv2.imread('assets/temp/plotimage.png')
            # graphimg = cv2.resize(graphimg, (430, 320))

            # GPAH END

            ret, img = cap.read()

            if ret is True:

                image = cv2.resize(img, (resize_val, sf))

                ROI_mask = np.zeros_like(image)
                cv2.circle(ROI_mask, (circle_center[0], circle_center[1]), circle_radius * 2, (255, 255, 255), -1)

                ROI_result = cv2.bitwise_and(ROI_mask, image)

                hsv = cv2.cvtColor(ROI_result, cv2.COLOR_BGR2HSV)

                # Define the lower and upper bounds for yellow color in HSV
                # lower_yellow = np.array([20, 100, 100])
                # upper_yellow = np.array([30, 255, 255])

                # Create a mask to extract yellow regions
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                # cv2.imshow('img', mask)
                # Apply some morphological operations to refine the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the contour corresponding to the yellow lane
                max_area = 0
                largest_contour = None
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        largest_contour = contour

                # Find the bottommost point of the largest contour
                if largest_contour is not None:
                    bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                else:
                    print(" ")

                # Display the bottommost point on the image
                if largest_contour is not None:
                    cv2.circle(image, bottommost, 5, (0, 255, 0), -1)
                    # cv2.imshow('Image with Bottommost Point', image)

                    point = bottommost
                    opacity = 0.5  # Opacity (25%)

                    # Calculate the distance between the point and the circle center
                    distance = math.sqrt((point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2)

                    # Create a blank image to visualize the circles and point

                    overlay = image.copy()

                    # Draw the circle and the point
                    cv2.circle(image, circle_center, circle_radius, (0, 255, 0), -1)  # Green circle for radius

                    # Create a transparent overlay
                    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

                    if distance > circle_radius:
                        wrong = wrong + 1
                        deviation.append(distance - 50)
                        image = cv2.putText(image, f"Deviation : {distance - 50:.2f}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.circle(image, point, 4, (0, 0, 255), -1)  # Blue circle for the point
                        cv2.line(image, point, circle_center, (0, 0, 255), 2)  # Green line between point1 and point2

                    if distance <= circle_radius:
                        correct = correct + 1
                        image = cv2.putText(image, f"Correct", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.circle(image, point, 4, (255, 0, 0), -1)  # Blue circle for the point
                        cv2.line(image, point, circle_center, (255, 0, 0), 2)  # Green line between point1 and point2

                    # Show the image with circles and the point

                    # cv2.imshow('Circle and Point', image)
                    image = cv2.resize(image, (int(resize_val / 2), int(sf / 2)))

                img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(img1))

                g1 = cv2.cvtColor(graphimg, cv2.COLOR_BGR2RGB)
                g2 = ImageTk.PhotoImage(Image.fromarray(g1))

                embed['image'] = img
                embed2['image'] = g2
                e_ws.update()
            else:
                e_ws.destroy()

                print(f'''Driving Report
                        Date : {datetime.datetime.now().strftime("%Y-%m-%d")}
                        Name : {name}
                        Address : {address}
                        Phone : {phnno}
                        Deviation From the line : {str(int(np.median(deviation)))}''')

        cap.release()

    entry1 = Entry()
    entry1.place(x=118, y=237, width=330, height=35)
    entry1.configure(font="-family {Poppins} -size 12")
    entry1.configure(background="SystemButtonFace", highlightthickness=0)

    entry2 = Entry()
    entry2.place(x=118, y=319, width=330, height=35)
    entry2.configure(font="-family {Poppins} -size 12")
    entry2.configure(background="SystemButtonFace", highlightthickness=0)

    entry3 = Entry()
    entry3.place(x=118, y=408, width=330, height=72)
    entry3.configure(font="-family {Poppins} -size 12")
    entry3.configure(background="SystemButtonFace", highlightthickness=0)

    btn_3 = PhotoImage(file=r"assets\images\selectfile.png")
    button1 = Button()
    button1.place(x=403, y=530, width=38, height=35)
    button1.configure(cursor="hand2")
    button1.configure(image=btn_3)
    button1.configure(borderwidth="0")
    button1.configure(command=select_file)

    e_ws.mainloop()


global filename
global e_ws
global name
global address
global phnno

circle_center = None
main()
