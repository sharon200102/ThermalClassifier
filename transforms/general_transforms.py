from ..datasets.classes import Detections, ImageSample, Detection
import numpy as np
from typing import Tuple
from pybboxes import BoundingBox
import random
from torchvision import transforms
from torchvision.transforms.functional import resize


class DownSampleImage():
    def __init__(self, down_scale_factor) -> None:
        self.down_scale_factor = down_scale_factor

    def __call__(self, sample: ImageSample):
        # image size is [C, H, W] 
        scale_size = int(min(sample.image.shape[1:]) * self.down_scale_factor)
        sample.image = resize(sample.image, size=scale_size, antialias=False)
        if isinstance(sample.label, Detection):
            sample.label.bbox.scale(self.down_scale_factor ** 2)
        return sample

class ToTensor():
    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, sample: ImageSample):
        sample.image = self.transform(sample.image)
        return sample


class ChoseDetection():
    def __init__(self, class2idx: dict, deterministic: bool = False, allow_background: bool = True) -> None:
        self.class2idx = class2idx
        self.allow_background = allow_background

        if deterministic:
            np.random.seed(42)


    def __call__(self,sample:ImageSample):
        assert type(sample.label) == Detections
        # get all objects from the detections dict

        sample_existing_classes = set(sample.label.detections.keys())

        if self.allow_background:
            sample_existing_classes.add(self.class2idx['BACKGROUND'])

        selected_class = np.random.choice(list(sample_existing_classes))
        
        sample.metadata['chosed_class_name'] = list(self.class2idx.keys())[selected_class]
        sample.metadata['chosed_class_idx'] = selected_class

        # Chose a random detection from the instances of the selected class, if background was chosen return None
        random_detection_of_selected_class = np.random.choice(sample.label.detections.get(selected_class,[None]))
        sample.label = random_detection_of_selected_class
        
        return sample
    
class AddShape():
    def __call__(self,sample:ImageSample):
        metadata = sample.metadata
        _, metadata['H'], metadata['W'] = sample.image.shape
        return sample
    
class SelectCropCoordinates:
    def __init__(self, area_scale:Tuple[float,float] = (1.0, 1.0), 
                    ratio:Tuple[float,float] = (1, 2), deterministic: bool = False) -> None:
        self.area_scale = area_scale
        self.ratio = ratio

        if deterministic:
            np.random.seed(42)
            random.seed(42)

    def __call__(self, sample: ImageSample):
        W, H = sample.metadata["W"], sample.metadata["H"]
        min_w_crop_size, min_h_crop_size = 10, 10
        w_crop, h_crop = int(np.clip(np.random.exponential(30), min_w_crop_size, W)), int(np.clip(np.random.exponential(30), min_h_crop_size, H))
        possible_sampling_range_x = (0, W - w_crop + 1)
        possible_sampling_range_y = (0, H - h_crop + 1)

        if sample.metadata['chosed_class_name'] != 'BACKGROUND':
            assert type(sample.label) == Detection
            # Select an augmented crop round the existing detection
            detection: BoundingBox = sample.label.bbox
            w_crop, h_crop = self.generate_crop_dimensions(detection.area)
            x0_detection, y0_detection, w_detection, h_detection = detection.raw_values
            crop_w_larger = w_crop >= w_detection
            crop_h_larger = h_crop >= h_detection

            possible_sampling_range_x = (x0_detection - (w_crop - w_detection), x0_detection) if crop_w_larger \
                                        else (x0_detection, x0_detection + (w_detection - w_crop))
            possible_sampling_range_y = (y0_detection - (h_crop - h_detection), y0_detection) if crop_h_larger \
                                        else (y0_detection, y0_detection + (h_detection - h_crop))
            
            possible_sampling_range_x = np.clip(possible_sampling_range_x, 0, W - w_crop)
            possible_sampling_range_y = np.clip(possible_sampling_range_y, 0, H - h_crop)
        
        if len(set(possible_sampling_range_x)) > 1:
            x0 = np.random.randint(*possible_sampling_range_x)
        else:
            x0 = possible_sampling_range_x[0]

        if len(set(possible_sampling_range_y)) > 1:
            y0 = np.random.randint(*possible_sampling_range_y)
        else:
            y0 = possible_sampling_range_y[0]

        crop = BoundingBox.from_coco(x0, y0, w_crop, h_crop)
        sample.metadata['crop_coordinates'] = crop.to_voc().raw_values
        return sample

    
    def generate_crop_dimensions(self, area):
        area =  area * np.random.uniform(*self.area_scale)
        ratio = np.random.uniform(*self.ratio)
        w = int(np.sqrt(area) * np.sqrt(ratio))
        h = int(np.sqrt(area) / np.sqrt(ratio))

        return w, h

class CropImage():
    def __call__(self,sample:ImageSample):
        x0, y0, x1, y1 = sample.metadata['crop_coordinates']
        sample.image = sample.image[:, y0: y1, x0: x1]
        return sample

class DetectionToClassificaton():
    def __call__(self, sample:ImageSample):
    
        sample.label = sample.metadata['chosed_class_idx']
        return sample