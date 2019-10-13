from PIL import Image
from skimage import transform
import numpy as np


class Preprocess:
    def __init__(self, img):
        if type(img) is not np.ndarray:
            self.image = np.array(Image.open(img))

        self.process_image()

    def crop(self, crop_area=[150, -100, 50, -50]):
        self.image = self.image[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]]

    def normalize(self):
        self.image = self.image/255.0

    def resize(self):
        self.image = transform.resize(self.image, [175, 350, 3])

    def process_image(self):
        self.crop()
        self.normalize()
        self.resize()

    def display_img(self):
        disp_img = Image.fromarray(self.image)
        disp_img.show()

if __name__ == "__main__":
    img = Preprocess("test.png")
