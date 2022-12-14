from mrcnn.utils import Dataset
import json
import os
#import mrcnn.utils as utils
from PIL import Image, ImageDraw
import numpy as np

# class that defines and loads the kangaroo dataset
class CustomDataset(Dataset):
    # load the dataset definitions
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        #print(annotation_json)
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "onioncell"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            #print('class id {}: {}'.format(class_id, class_name))
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            #print('Annotation: image ID = {}'.format(image_id))
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            #print('Images: image ID = {}'.format(image_id))
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_file_name = image_file_name.split(os.path.sep)[-1]
                image_path = os.path.sep.join([images_dir, image_file_name])
                if image_id in annotations:
                    image_annotations = annotations[image_id]

                    # Add the image using the base method from utils.Dataset
                    self.add_image(
                        source=source_name,
                        image_id=image_id,
                        path=image_path,
                        width=image_width,
                        height=image_height,
                        annotations=image_annotations
                    )
                #else:
                #    print('Image ID {} has no annotations.'.format(image_id))
                
    # load the masks for an image
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            
            x0 = annotation['bbox'][0]
            y0 = annotation['bbox'][1]
            x1 = x0 + annotation['bbox'][2]
            y1 = y0 + annotation['bbox'][3]
            box_coord = [x0, y0, x1, y1]
            
            mask_draw.rectangle(box_coord, fill=1)
            bool_array = np.array(mask) > 0
            instance_masks.append(bool_array)
            class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids
    
    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
