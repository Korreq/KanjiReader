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

        # Sort the bounding boxes from right to left and then top to bottom
        bounding_boxes = self.sort_bounding_boxes(bounding_boxes)

        return bounding_boxes

    def merge_overlapping_boxes(self, bounding_boxes):
        """Merges overlapping and encapsulated bounding boxes.""" 
        while True: 
            merged_boxes = [] 
            merge_occurred = False 
            
            for box in bounding_boxes: 
                points = np.array(box, dtype=np.int32) 
                found_overlap = False 
                
                for i, merged_box in enumerate(merged_boxes): 
                    merged_points = np.array(merged_box, dtype=np.int32) 
                    
                    if self.is_overlapping(points, merged_points) or self.is_contained(points, merged_points): 
                        merged_box = self.merge_two_boxes(points, merged_points) 
                        merged_boxes[i] = merged_box 
                        found_overlap = True 
                        merge_occurred = True 
                        break 
                    
                if not found_overlap: 
                    merged_boxes.append(box) 
            
            bounding_boxes = merged_boxes 

            if not merge_occurred: 
                break 
            
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
            if cv2.pointPolygonTest(box2_points, (float(point[0]), float(point[1])), measureDist=False) < 0:
                return False

        return True

    def merge_two_boxes(self, box1, box2):
        """Merge two overlapping or contained boxes into one."""
        all_points = np.concatenate((box1, box2), axis=0)  # Combine the points of both boxes
        merged_rect = cv2.minAreaRect(all_points)  # Get the minimum bounding rectangle of the combined points
        merged_box = cv2.boxPoints(merged_rect)  # Get the corner points of the merged rectangle
        merged_box = np.array(merged_box, dtype=np.int32)  # Convert points to integer (using np.int32)
        return merged_box

    def sort_bounding_boxes(self, bounding_boxes): 
        """Sorts bounding boxes from right to left and then top to bottom.""" 
        # Calculate the centroid of each bounding box 
        centroids = [np.mean(box, axis=0) for box in bounding_boxes] 
        
        # Sort primarily by the x-coordinate (right to left) and secondarily by the y-coordinate (top to bottom) 
        sorted_indices = sorted(range(len(centroids)), key=lambda i: (-centroids[i][0], centroids[i][1]))
        
        return [bounding_boxes[i] for i in sorted_indices]
 
    def save_image_with_bounding_boxes(self, output_filename: str, bounding_boxes):
        """Draw bounding boxes on the image and save."""

        for box in bounding_boxes:
            # Unpack the four corners of the rotated bounding box
            points = np.array(box, dtype=np.int32)
            
            # Reshape to (N, 1, 2) format for polylines
            points = points.reshape((-1, 1, 2))  
            
            # Draw the rotated rectangle using polylines
            cv2.polylines(self.image, [points], isClosed=True, color=(0, 255, 0), thickness=2) 

        cv2.imwrite(output_filename, self.image)
        print(f"Image saved as: {output_filename}")

        