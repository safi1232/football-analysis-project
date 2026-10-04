import cv2
from ultralytics import YOLO


video_path = r"C:\Users\Abid\3D Objects\fotball analysis\video\08fd33_4.mp4"
model_path = r"C:\Users\Abid\3D Objects\fotball analysis\Models\best.pt"

# Load video
cap = cv2.VideoCapture(video_path)

# Read ONLY first frame
ret, frame = cap.read()

cap.release()

if not ret:
    print("Could not read first frame.")
    exit()

# Load YOLO
model = YOLO(model_path)

# Detect on ONLY first frame
results = model.predict(
    frame,
    conf=0.1,
    verbose=False
)

result = results[0]

# Find players
for i, class_id in enumerate(result.boxes.cls):

    class_id = int(class_id)

    class_name = result.names[class_id]

    if class_name == "player":

        bbox = result.boxes.xyxy[i].tolist()

        x1, y1, x2, y2 = map(int, bbox)

        # Crop player
        cropped_image = frame[y1:y2, x1:x2]

        # Save
        cv2.imwrite(
            r"C:\Users\Abid\3D Objects\fotball analysis\outputvideos\cropped_player.jpg",
            cropped_image
        )

        print("Player cropped successfully!")
        print("Bounding box:", bbox)

        break