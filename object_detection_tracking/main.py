import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection + tracking (ByteTrack)
    results = model.track(frame, persist=True, conf=0.5)

    for r in results:
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} | ID {int(track_id)}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("YOLOv8 Multi-Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




