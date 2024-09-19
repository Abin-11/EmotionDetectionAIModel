from django.http import JsonResponse
import cv2
from deepface import DeepFace
import requests
import mediapipe as mp
import mysql.connector
import numpy as np

# Initialize Mediapipe Face Detection and Drawing
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Emotion to intent mapping
emotion_to_intent = {
    "happy": "HappyUserIntent",
    "sad": "ComfortUserIntent",
    "angry": "CalmUserIntent",
    "surprise": "ExplainMoreIntent",
    "fear": "ReassureUserIntent",
    "neutral": "NeutralUserIntent"
}

# Voiceflow API Key and Endpoint
VOICEFLOW_API_KEY = "VF.DM.66d06ebdfed942a330ae909e.aICoVrrFaHDTfLqx"
VOICEFLOW_API_ENDPOINT = "https://creator.voiceflow.com/prototype/66cf559b5f47d6b5d0baeff1"

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'your_mysql_user',
    'password': 'your_mysql_password',
    'database': 'your_database'
}

# Function to get registered faces from the database
def get_registered_faces():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, face_encoding FROM registered_faces")
    faces = cursor.fetchall()

    conn.close()
    return faces

# Check if face is registered
def is_face_registered(face_encoding, registered_faces):
    for user_id, stored_encoding in registered_faces:
        stored_encoding = np.fromstring(stored_encoding, sep=',')
        similarity = np.dot(stored_encoding, face_encoding) / (np.linalg.norm(stored_encoding) * np.linalg.norm(face_encoding))
        if similarity > 0.9:
            return user_id
    return None

# Detect faces using Mediapipe and DeepFace
def detect_faces():
    cap = cv2.VideoCapture(0)
    registered_faces = get_registered_faces()
    user_id = None

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                    face_image = frame[y:y + height, x:x + width]

                    try:
                        face_encoding = DeepFace.represent(face_image, model_name="Facenet")[0]['embedding']
                        user_id = is_face_registered(face_encoding, registered_faces)

                        if user_id:
                            print(f"Face is registered as user_id: {user_id}")
                            break
                        else:
                            print("Face not registered")
                    except Exception as e:
                        print("Error during face encoding:", e)

            if user_id:
                break

    cap.release()
    cv2.destroyAllWindows()

    return user_id

# Detect emotions using DeepFace
def detect_emotions():
    cap = cv2.VideoCapture(0)
    emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'])
            emotion = analysis['dominant_emotion']
            print(f"Detected Emotion is: {emotion}")
            break
        except Exception as e:
            print("Error", e)
            break

    cap.release()
    cv2.destroyAllWindows()

    if emotion is None:
        emotion = "neutral"
    return emotion

# Map detected emotion to the corresponding intent
def map_emotion_to_intent(emotion):
    return emotion_to_intent.get(emotion.lower(), "DefaultIntent")

# Call the Voiceflow API
def voiceflow_api(user_id, intent):
    headers = {
        "Authorization": VOICEFLOW_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "request": {
            "type": "text",
            "payload": intent
        },
        "state": {}
    }

    response = requests.post(f"{VOICEFLOW_API_ENDPOINT}/user/{user_id}", json=data, headers=headers)

    if response.status_code == 200:
        return response.json().get("trace", [])
    else:
        print(f"Error calling Voiceflow API: {response.status_code}, {response.text}")
        return []

# Django view to handle intent detection
def detect_intent(request):
    detected_emotion = detect_emotions()
    chatbot_intent = map_emotion_to_intent(detected_emotion)

    user_id = "123"  # Hardcoded user_id, replace with dynamic if needed
    voiceflow_response = voiceflow_api(user_id, chatbot_intent)

    return JsonResponse({
        "emotion": detected_emotion,
        "intent": chatbot_intent,
        "voiceflow_response": voiceflow_response
    })

