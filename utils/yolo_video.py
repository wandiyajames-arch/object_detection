import cv2
import csv
import time

# Inside your Detector class's forward() method or main video processing loop:
def process_video(self):
    cap = cv2.VideoCapture(self.filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # Open a CSV file to store the logs
    with open('tracking_logs.csv', mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        # Write the header based on project requirements
        log_writer.writerow(['timestamp', 'frame', 'track_id', 'class', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = round(frame_count / fps, 2)
            
            # Run your YOLO tracking here (assuming 'results' holds the output)
            # results = model.track(frame, persist=True, ...)
            
            object_present = False
            
            # Check if detections exist in the current frame
            if results[0].boxes.id is not None:
                object_present = True
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    class_name = self.model.names[class_id]
                    
                    # 1. Log the data
                    log_writer.writerow([
                        timestamp, frame_count, track_id, class_name,
                        int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    ])
                    
                    # (Your existing code to draw bounding boxes and IDs goes here)
            
            # 2. Add the empty scene indicator
            if not object_present:
                cv2.putText(frame, "NO SELECTED OBJECTS PRESENT", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()