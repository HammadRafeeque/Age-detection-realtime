import cv2
import numpy as np

# Model files
face_model = "opencv_face_detector_uint8.pb"
face_proto = "opencv_face_detector.pbtxt"
age_model = "age_net.caffemodel"
age_proto = "age_deploy.prototxt"

# Mean values and age buckets
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load networks
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

# Function to detect faces
def detect_faces(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                 [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2))
    return boxes

# Start webcam
cap = cv2.VideoCapture(0)
padding = 20

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_boxes = detect_faces(frame)
    for (x1, y1, x2, y2) in face_boxes:
        # Add padding and clip to image boundaries
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(frame.shape[1] - 1, x2 + padding)
        y2_p = min(frame.shape[0] - 1, y2 + padding)

        face = frame[y1_p:y2_p, x1_p:x2_p]

        # Prepare blob and predict age
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                     MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Draw box and label
        label = f"Age: {age}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Real-Time Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
