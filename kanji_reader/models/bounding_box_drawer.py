# bounding_box_drawer.py
import cv2
import easyocr
import numpy as np

class BoundingBoxDrawer:
    def __init__(self, image_path: str):
        """Initializes the BoundingBoxDrawer with the image file."""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.reader = easyocr.Reader(['ja'])  # Initialize easyocr for Japanese text recognition

    def detect_text_regions(self):
        """Detect text regions (bounding boxes) using easyOCR."""
        # Use easyOCR to detect text in the image
        result = self.reader.readtext(self.image)
        bounding_boxes = []

        # Extract bounding boxes from the result
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            bounding_boxes.append([top_left, top_right, bottom_right, bottom_left])

        # Merge overlapping boxes based on axis-aligned bounding rectangles
        bounding_boxes = self.merge_overlapping_boxes(bounding_boxes)

        return bounding_boxes

    def merge_overlapping_boxes(self, bounding_boxes):
        """Merges overlapping bounding boxes."""
        merged_boxes = []
        
        for box in bounding_boxes:
            points = np.array(box, dtype=np.int32) 
            found_overlap = False 
        
            for i, merged_box in enumerate(merged_boxes): 
                merged_points = np.array(merged_box, dtype=np.int32) 
                
                if self.is_overlapping(points, merged_points) or self.is_contained(points, merged_points): 
                    merged_box = self.merge_two_boxes(points, merged_points) 
                    merged_boxes[i] = merged_box 
                    found_overlap = True 
                    break

            if not found_overlap: 
                merged_boxes.append(box) 

        return merged_boxes

    def is_overlapping(self, box1, box2):
        """Check if two bounding boxes overlap."""
        # Use OpenCV's polygon intersection or custom algorithm for checking intersection
        poly1 = np.array(box1) 
        poly2 = np.array(box2) 
        return cv2.rotatedRectangleIntersection((cv2.minAreaRect(poly1)), (cv2.minAreaRect(poly2)))[0] != 0

    def is_contained(self, box1, box2):
        """Check if one bounding box is partially or fully contained within another."""
        rect2 = cv2.minAreaRect(np.array(box2))
        box2_points = cv2.boxPoints(rect2)

        # Check if any point of box1 is inside box2
        for point in box1:
            if cv2.pointPolygonTest(box2_points, (float(point[0]), float(point[1])), measureDist=False) >= 0:
                return True

        return False

    def merge_two_boxes(self, box1, box2):
        """Merge two overlapping or contained boxes into one."""
        all_points = np.concatenate((box1, box2), axis=0)  # Combine the points of both boxes
        merged_rect = cv2.minAreaRect(all_points)  # Get the minimum bounding rectangle of the combined points
        merged_box = cv2.boxPoints(merged_rect)  # Get the corner points of the merged rectangle
        merged_box = np.array(merged_box, dtype=np.int32)  # Convert points to integer (using np.int32)
        return merged_box

    def draw_bounding_boxes(self, bounding_boxes):
        """Draw bounding boxes on the image."""
        for box in bounding_boxes:
            # Unpack the four corners of the rotated bounding box
            points = np.array(box, dtype=np.int32)
            
            # Reshape to (N, 1, 2) format for polylines
            points = points.reshape((-1, 1, 2))  
            
            # Draw the rotated rectangle using polylines
            cv2.polylines(self.image, [points], isClosed=True, color=(0, 255, 0), thickness=2) 
            
        return self.image
    
    def save_image(self, output_filename: str):
        """Save the image with bounding boxes to a file."""
        cv2.imwrite(output_filename, self.image)
        print(f"Image saved as: {output_filename}")