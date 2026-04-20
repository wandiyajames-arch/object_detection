import cv2 as cv

class SSDDetector():
    def __init__(self, filepath):
        self.filepath = filepath
        self.prototxt  = "models/ssd_mobilenet.prototxt"
        self.caffemodel = "models/ssd_mobilenet.caffemodel"
        self.conf_threshold = 0.5
        self.CLASSES = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
        ]

    def forward(self):
        net = cv.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        cap = cv.VideoCapture(self.filepath)
        assert cap.isOpened(), "Issue reading the video"

        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        writer = cv.VideoWriter("ssd_detection.avi", cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < self.conf_threshold:
                    continue
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                label = f"{self.CLASSES[idx]}: {confidence:.2f}"
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, label, (x1, y1 - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            writer.write(frame)
            cv.imshow("SSD Detection", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        writer.release()
        cv.destroyAllWindows()