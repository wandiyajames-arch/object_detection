from utils.cascade import Classifier
from utils.yolo_video import Detector
from utils.watershed import WatershedClassifier
from utils.ssd import SSDDetector
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Object detection")
    parser.add_argument("--model", type=str,
                        choices=['cascade', 'yolov8', 'yolov11', 'watershed', 'ssd'],
                        default='cascade', help="Model type")
    parser.add_argument("--filepath", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.model == 'cascade':
        my_classifier = Classifier(args.filepath)
        my_classifier.plot()
    elif args.model == 'yolov11':
        my_detector = Detector(args.filepath)
        my_detector.forward()
    elif args.model == 'watershed':
        my_classifier = WatershedClassifier(args.filepath)
        my_classifier.plot()
    elif args.model == 'ssd':
        my_detector = SSDDetector(args.filepath)
        my_detector.forward()

if __name__ == '__main__':
    main()