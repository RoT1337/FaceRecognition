import pickle
import face_recognition
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox

import SaveFace

def register_old_picture():
    messagebox.showinfo("Welcome", "Select a picture to encode as a pickle file")
    name = SaveFace.get_user_name()
    if (not name):
        print("No name entered.")
        return
    
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        print("No file selected. Exiting...")
        return

    # Load the image and extract the face encoding
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

        # Save the encoding to a pickle file
        encoding_file = f"FaceRecognition/encodings/{name}.pkl"
        with open(encoding_file, "wb") as file:
            pickle.dump(face_encoding, file)
        print(f"Face encoding saved for {name} at {encoding_file}")
        messagebox.showinfo("Success", f"Face encoding saved for {name}!")
    else:
        print("No face detected in the selected image.")
        messagebox.showerror("Error", "No face detected in the selected image.")

if __name__ == "__main__":
    register_old_picture()