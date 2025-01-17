from transformers import AutoModel, AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from PIL import Image
import cv2
import numpy as np
from .bounding_box_drawer import BoundingBoxDrawer


class OCRModels:
    def __init__(self):
        # Initialize OCR models for Manga and GOT OCR
        self.manga_ocr_model = None
        self.manga_ocr_tokenizer = None
        self.manga_ocr_processor = None
        self.got_model = None
        self.got_tokenizer = None

    def load_manga_ocr(self):
        """Load Manga OCR model if not already loaded."""
        if not self.manga_ocr_model:
            self.manga_ocr_processor = ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base")
            self.manga_ocr_model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
            self.manga_ocr_tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
        return self.manga_ocr_processor, self.manga_ocr_model, self.manga_ocr_tokenizer

    def load_got_ocr(self):
        """Load GOT OCR model if not already loaded."""
        if not self.got_model:
            self.got_model = AutoModel.from_pretrained("srimanth-d/GOT_CPU")
            self.got_tokenizer = AutoTokenizer.from_pretrained("srimanth-d/GOT_CPU")
            self.got_model.config.use_safetensor = True
            self.got_model.config.pad_token_id = self.got_tokenizer.eos_token_id
        return self.got_model, self.got_tokenizer

    def text_from_image_manga_ocr(self, image_path: str) -> str:
        """Extract text from an image using Manga OCR."""
        image_processor, model, tokenizer = self.load_manga_ocr()

        bbox_drawer = BoundingBoxDrawer(image_path)
        bounding_boxes = bbox_drawer.detect_text_regions()

        # Draw the bounding boxes and save the image
        image_with_bboxes = bbox_drawer.draw_bounding_boxes(bounding_boxes)
        output_filename = f"{image_path.split('.')[0]}_with_bounding_boxes.png"
        bbox_drawer.save_image(output_filename)

        # Extract text from the bounding boxes
        recognized_text = self.extract_text_from_bboxes(image_path, bounding_boxes, model, image_processor, tokenizer)
        
        return recognized_text

    def extract_text_from_bboxes(self, image_path: str, bounding_boxes, model, image_processor, tokenizer):
        """Use OCR model to recognize text within bounding boxes."""
        recognized_texts = []
        # Load the original image using OpenCV (for cropping)
        bounding_image = cv2.imread(image_path)

        for box in bounding_boxes:
            points = np.array(box, dtype=np.int32)
            rect = cv2.minAreaRect(points)  # Get the rotated rectangle for the box
            box = cv2.boxPoints(rect)  # Get the four points of the rotated box
            box = np.array(box, dtype=np.int32)  # Convert the points to integers
            
            # Get the bounding box area
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)

            # Ensure that the bounding box area is valid
            if x_min >= x_max or y_min >= y_max:
                print(f"Skipping invalid bounding box: {box}")
                continue

            # Crop the image based on the rotated bounding box
            cropped_image = bounding_image[y_min:y_max, x_min:x_max]
            
            # Skip empty cropped images
            if cropped_image.size == 0:
                print(f"Skipping empty cropped image at box: {box}")
                continue
    
            
            # Convert cropped image to PIL for OCR model
            pil_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            
            # Preprocess the cropped image for the OCR model
            inputs = image_processor(images=pil_cropped_image, return_tensors="pt").pixel_values
            outputs = model.generate(inputs)

            pil_cropped_image.close()

            # Decode the output text
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ","")
            recognized_texts.append(text)
        
        return " ".join(recognized_texts)


    def text_from_image_got(self, picture: str) -> str:
        """Extract text from an image using srimanth-d/GOT_CPU."""    
        model, tokenizer = self.load_got_ocr()

        return model.chat(tokenizer, picture, ocr_type='ocr')