# Step 1: Import the necessary library
import cv2

# Step 2: Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 3: Access the webcam
cap = cv2.VideoCapture(0)  # 0 means the default webcam

# Step 4: Start reading frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 5: Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Step 6: Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangles

    # Step 7: Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Step 8: Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
