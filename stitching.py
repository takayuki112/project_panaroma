import numpy 
import cv2
import os

class Stitcher():
    def __init__(self, input_dir, output_dir, feature_detector="SIFT"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.input_images = []
        
        # Feature Detector
        if feature_detector == "SIFT":
            self.feature_detector = cv2.SIFT_create()
        elif feature_detector == "ORB":
            self.feature_detector = cv2.ORB_create()
        else:
            raise ValueError("Invalid feature detector. Use 'SIFT' or 'ORB'.")
        
        self.feature_points_and_descriptors = []
        
        
    def read_input_dir(self):
        image_extensions = {".jpeg", ".jpg", ".png"}
        
        for file in os.listdir(self.input_dir):
            if os.path.splitext(file)[1] in image_extensions:
                img = cv2.imread(os.path.join(self.input_dir, file))
                if img is not None:
                    self.input_images.append(img)
                else:
                    print(f"Could not read image {file} in {self.input_dir}")
        
        if len(self.input_images) < 2:
            raise ValueError("Not enough images in the input directory.")
        else:
            print(f"Found {len(self.input_images)} images in the input directory.")
            
    
    def detect_keypoints_and_descriptors(self):
        
        for i, img in enumerate(self.input_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            self.feature_points_and_descriptors.append((keypoints, descriptors))
            
            if keypoints is not None and descriptors is not None:
                self.keypoints_and_descriptors.append((keypoints, descriptors))
                print(f"Detected {len(keypoints)} keypoints in image {i+1}")
            else:
                print(f"Failed to detect keypoints in image {i+1}")

        print("Feature detection completed.")
    
    

