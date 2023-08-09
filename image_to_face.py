from PIL import Image
from numpy import asarray
from mtcnn_old import MTCNN
import os
import cv2
import time

# Method to extract and save face images from live video for a given duration
def extract_and_save_faces_from_video(output_directory, name, duration):
    detector = MTCNN()
    video_capture = cv2.VideoCapture(0)  # Use 0 for default camera, you can change this if you have multiple cameras

    os.makedirs(os.path.join(output_directory, name), exist_ok=True)

    frame_count = 0
    start_time = time.time()
    while (time.time() - start_time) <= duration:
        ret, frame = video_capture.read()
        if not ret:
            break

        pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect(pixels)
        if len(faces) > 0:
            x1, y1, w, h = faces[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = abs(x1 + w), abs(y1 + h)
            face_region = pixels[y1:y2, x1:x2]
            face_img = Image.fromarray(face_region, 'RGB')
            face_img = face_img.resize((112, 112))

            output_path = os.path.join(output_directory, name, f"image{frame_count}.jpg")
            face_img.save(output_path)
            print(f"Face saved: {output_path}")

            frame_count += 1

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Define paths for processed data
output_data_path = "data/processed"
F_output_data_path = "data/facebank"

# Get user input for the name and duration (in seconds)
name = input("Enter the name of the person: ")
duration = 20

# Extract and save faces from live video for the specified duration
extract_and_save_faces_from_video(output_data_path, name, duration)
extract_and_save_faces_from_video(F_output_data_path, name, duration)
