import io
import pickle
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://celebrecognize-default-rtdb.firebaseio.com/",
    'storageBucket': "celebrecognize.appspot.com"
})

bucket = storage.bucket()

# Load the encoding file
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, celebIds = encodeListKnownWithIds

modeType = 0
counter = 0
id = -1
imgceleb = []


def recognizeFaces(SSAddress):
    arrID = []
    blob = bucket.blob(str(SSAddress))
    image_data = blob.download_as_bytes()


    # Now you can use the image data with your model
    # For example, using Pillow (PIL) to open the image from memory
    imageSS = io.BytesIO(image_data)

    unknown_image = face_recognition.load_image_file(imageSS)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        best_match_index = np.argmin(face_distances)
        celebId = celebIds[best_match_index]
        arrID.append(celebId)
        # print(celebId)
    return arrID
        # celebInfo = db.reference(f'Celebs/{celebId}').get()
        # print(celebInfo)

        # Get the Image from the storage
        # imgceleb_blob = bucket.get_blob(f'celebPics/{celebId}.jpg')
        # if imgceleb_blob:
        #     imgceleb_array = np.frombuffer(imgceleb_blob.download_as_string(), np.uint8)
        #     imgceleb = cv2.imdecode(imgceleb_array, cv2.COLOR_BGRA2BGR)
        #     imgceleb = cv2.resize(imgceleb, (512, 675))

        #     if imgceleb is not None:
        #         cv2.imshow('Celeb Image', imgceleb)
        #         cv2.waitKey(0)  # Wait for a key press
        #         cv2.destroyAllWindows()  # Close the image window
        #     else:
        #         print("No celeb image found.")
        # else:
        #     print(f"No celeb image found for ID: {celebId}")
