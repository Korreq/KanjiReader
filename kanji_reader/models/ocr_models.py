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
            self.got_tokenizer = AutoTokenizer.from_pretrained("srimanth-d/GOT_CPU", trust_remote_code=True)
            self.got_model = AutoModel.from_pretrained(
                "srimanth-d/GOT_CPU", trust_remote_code=True, 
                low_cpu_mem_usage=True, use_safetensors=True, 
                pad_token_id=self.got_tokenizer.eos_token_id
            )
        return self.got_model, self.got_tokenizer

    def text_from_image_manga_ocr(self, image_path: str) -> str:
        """Extract text from an image using Manga OCR."""
        recognized_texts = []
        image_processor, model, tokenizer = self.load_manga_ocr()

        bbox_drawer = BoundingBoxDrawer(image_path)
        bounding_boxes = bbox_drawer.detect_text_regions()

        # Draw the bounding boxes and save the image
        output_filename = f"{image_path.split('.')[0]}_with_bounding_boxes.png"
        bbox_drawer.save_image_with_bounding_boxes( output_filename, bounding_boxes )

        # Get cropped images from the bounding boxes
        cropped_images = self.crop_image_from_bounding_box(image_path, bounding_boxes)

        for cropped_image in cropped_images:

            # Preprocess the cropped image for the OCR model
            inputs = image_processor(images=cropped_image, return_tensors="pt").pixel_values
            outputs = model.generate(inputs)

            cropped_image.close()

            # Decode the output text
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ","")
            recognized_texts.append(text)

        return " ".join(recognized_texts)

    def text_from_image_got(self, image_path: str) -> str: 
        """Extract text from an image using GOT OCR.""" 
        recognized_texts = []
        got_model, got_tokenizer = self.load_got_ocr() 
        
        bbox_drawer = BoundingBoxDrawer(image_path) 
        bounding_boxes = bbox_drawer.detect_text_regions() 

        # Draw the bounding boxes and save the image 
        output_filename = f"{image_path.split('.')[0]}_with_bounding_boxes.png" 
        bbox_drawer.save_image_with_bounding_boxes( output_filename, bounding_boxes )

        # Get cropped images from the bounding boxes 
        cropped_images = self.crop_image_from_bounding_box(image_path, bounding_boxes)

        for cropped_image in cropped_images:

            temp_filename = "data/images/temp/temp_cropped_image.png" 
            cropped_image.save(temp_filename) 
                
            text = got_model.chat(got_tokenizer, temp_filename, ocr_type='ocr').replace("\n","") 
            recognized_texts.append(text)    

            cropped_image.close()

        return " ".join(recognized_texts)

    def crop_image_from_bounding_box(self, image_path, bounding_boxes):
        """Crop images based on bounding boxes and their rotations."""
        cropped_images = []
        bounding_image = cv2.imread(image_path)

        for box in bounding_boxes:
            points = np.array(box, dtype=np.int32) # Convert the points to integers
            rect = cv2.minAreaRect(points)  # Get the rotated rectangle for the box
            box = cv2.boxPoints(rect)  # Get the four points of the rotated box
            box = np.array(box, dtype=np.int32)  # Convert the points to integers
            angle = rect[2]  # Get the angle of the bounding box
          

            # If the angle is close to 0 (straight box), use direct cropping
            if abs(angle) < 10 or abs(angle - 90) < 10:
                x_min, y_min = np.min(box, axis=0)
                x_max, y_max = np.max(box, axis=0)
                cropped_image = bounding_image[y_min:y_max, x_min:x_max]

            # If not, use perspective transform
            else:
                width = int(rect[1][0])
                height = int(rect[1][1])

                # Ensure width and height are positive
                if width < 0:
                    width = -width
                if height < 0:
                    height = -height

                # Define the destination points for the warp transform (straight rectangle)
                dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

                # Calculate the perspective transform matrix and apply it
                M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
                cropped_image = cv2.warpPerspective(bounding_image, M, (width, height))

            if cropped_image.size == 0:
                print(f"Skipping empty cropped image at box:\n{box}")
                continue

            # Convert cropped image to PIL for OCR model
            pil_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            cropped_images.append(pil_cropped_image)

        return cropped_images