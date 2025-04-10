import face_recognition
import os, sys
import cv2
import numpy as np
import math
import tkinter.messagebox as messagebox

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_value = (1.0 - face_distance) / (range * 2.0)

    if (face_distance > face_match_threshold):
        return str(round(linear_value * 100, 2)) + '%'
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []

    known_face_encodings = []
    known_face_names = []

    process_current_frame = 0

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('FaceRecognition/faces'):
            face_image = face_recognition.load_image_file(f'FaceRecognition/faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        
        print(self.known_face_names)

    def run_recognition(self):

        webcam = cv2.VideoCapture(0)
        if (not webcam):
            sys.exit('Video source not found.')

        while True:
            ret, frame = webcam.read()
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if (self.process_current_frame % 4 == 0):
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if (matches[best_match_index]):
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    
                    self.face_names.append(f'{name} ({confidence})')
            self.process_current_frame += 1

            for face_landmarks in face_landmarks_list:
                for facial_feature in face_landmarks.keys():
                    for point in face_landmarks[facial_feature]:
                        point = (point[0] * 4, point[1] * 4)
                        cv2.circle(frame, point, 1, (255, 0, 255), -1)

                if 'chin' in face_landmarks:
                    chin_points = face_landmarks['chin']
                    middle_chin = chin_points[len(chin_points) // 2]
                    x, y = middle_chin[0] * 4, middle_chin[1] * 4
                    cv2.putText(frame, f'{name} ({confidence})', (x, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition with Landmarks and Confidence', frame)

            if cv2.waitKey(1) == ord('q'):
                messagebox.showinfo("Closing", "Closing the application, press OK")
                break

        webcam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()