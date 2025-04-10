import cv2
import tkinter as tk
from tkinter import simpledialog
import tkinter.messagebox as messagebox
import os

def get_user_name():
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter your name:")
    return name

def capture_picture():
    index = 0

    name = get_user_name()
    if (not name):
        print("No name entered. Exiting...")
        return
    
    messagebox.showinfo("Welcome", f"Hello {name}, position yourself in front of the camera and press 's' to take a picture.")

    save_dir = "FaceRecognition/faces"
    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if (not ret):
            print("Failed to access the webcam or webcam was closed.")
            break

        frame = cv2.flip(frame, 1)

        cv2.imshow("Webcam - Press 's' to take a picture, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if (key == ord('s')):
            file_name = os.path.join(save_dir, f"{name}_{index}.jpg")
            cv2.imwrite(file_name, frame)
            messagebox.showinfo("Successful", f"Picture saved as {file_name}")
            index += 1
        elif (key == ord('q')):
            messagebox.showinfo("Error", "Closing the application, press OK")
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_picture()