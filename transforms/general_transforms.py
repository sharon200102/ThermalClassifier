from ThermalClassifier.datasets.classes import BboxSample
import numpy as np
from typing import Tuple
from pybboxes import BoundingBox
import random
from torchvision import transforms
from torchvision.transforms.functional import resize, hflip, vflip, rotate


class RandomDownSampleImage():
    def __init__(self, down_scale_factor_range: list , p: float = 0.3) -> None:
        self.down_scale_factor_range = down_scale_factor_range
        self.p = p

    def __call__(self, sample: BboxSample):
        # image size is [C, H, W]
        if random.random() < self.p:
            down_scale_factor = random.uniform(*self.down_scale_factor_range)
            scale_size = int(min(sample.image.shape[1:]) * down_scale_factor)
            sample.image = resize(sample.image, size=scale_size, antialias=False)
            # if isinstance(sample.bbox, BoundingBox):
            sample.bbox.scale(down_scale_factor ** 2)
        return sample

class ToTensor():
    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, sample: BboxSample):
        sample.image = self.transform(sample.image)
        return sample

class RandomHorizontalFlip():
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, sample: BboxSample):
        if random.random() < self.p:
            sample.image = hflip(sample.image)
        return sample

class RandomVerticalFlip():
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, sample: BboxSample):
        if random.random() < self.p:
            sample.image = vflip(sample.image)
        return sample

class RandomRotation():
    def __init__(self, degrees: tuple) -> None:
        self.degrees = degrees

    def __call__(self, sample: BboxSample):
        angle = np.random.randint(*self.degrees)
        sample.image = rotate(sample.image, angle)
        return sample

class SampleBackground():
    def __init__(self, class2idx: dict, deterministic: bool = False, p: float = 0.3) -> None:
        self.class2idx = class2idx
        self.p = p

        if deterministic:
            np.random.seed(42)

    def __call__(self, sample: BboxSample):
        if random.random() < self.p:
            sample.label = self.class2idx['BACKGROUND']
        
        return sample
    
class AddShape():
    def __call__(self,sample:BboxSample):
        metadata = sample.metadata
        _, metadata['H'], metadata['W'] = sample.image.shape
        return sample
    
class SelectCropCoordinates:
    def __init__(self, class2idx: dict, area_scale: Tuple[float,float] = (1.0, 1.0), 
                    ratio:Tuple[float,float] = (1, 2), deterministic: bool = False) -> None:
        self.class2idx = class2idx
        self.area_scale = area_scale
        self.ratio = ratio

        if deterministic:
            np.random.seed(42)
            random.seed(42)

    def __call__(self, sample: BboxSample):
        W, H = sample.metadata["W"], sample.metadata["H"]
        w_crop, h_crop = self.generate_crop_dimensions(sample.bbox.area, (W, H))
        possible_sampling_range_x = (0, W - w_crop + 1)
        possible_sampling_range_y = (0, H - h_crop + 1)

        if sample.label != self.class2idx['BACKGROUND']:
            # Select an augmented crop round the existing detection
            # w_crop, h_crop = self.generate_crop_dimensions(sample.bbox.area)
            x0_detection, y0_detection, w_detection, h_detection = sample.bbox.raw_values
            crop_w_larger = w_crop >= w_detection
            crop_h_larger = h_crop >= h_detection

            possible_sampling_range_x = (x0_detection - (w_crop - w_detection), x0_detection) if crop_w_larger \
                                        else (x0_detection, x0_detection + (w_detection - w_crop))
            possible_sampling_range_y = (y0_detection - (h_crop - h_detection), y0_detection) if crop_h_larger \
                                        else (y0_detection, y0_detection + (h_detection - h_crop))
            
            possible_sampling_range_x = np.clip(possible_sampling_range_x, 0, W - w_crop).astype(int)
            possible_sampling_range_y = np.clip(possible_sampling_range_y, 0, H - h_crop).astype(int)
        
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

    
    def generate_crop_dimensions(self, area, image_size):
        image_w, image_h = image_size
        area =  area * np.random.uniform(*self.area_scale)
        ratio = np.random.uniform(*self.ratio)
        w = int(np.sqrt(area) * np.sqrt(ratio))
        h = int(np.sqrt(area) / np.sqrt(ratio))

        w = min(w, image_w)
        h = min(h, image_h)
        return w, h

class CropImage():
    def __call__(self,sample:BboxSample):
        x0, y0, x1, y1 = sample.metadata['crop_coordinates']
        sample.image = sample.image[:, y0: y1, x0: x1]
        return sample