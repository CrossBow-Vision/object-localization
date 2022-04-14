import glob
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from train import create_model, IMAGE_SIZE
from keras.applications.mobilenetv2 import preprocess_input

WEIGHTS_FILE = "model-0.52.h5"
IMAGES = "images_validation/*png"

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        
        unscaled_img = cv2.imread(filename)        
        img_scaled = preprocess_input(np.array(unscaled_img, dtype=np.float32))
        
        region = model.predict(x=np.array([img_scaled]))[0]

        x0 = int(region[0] )
        y0 = int(region[1] )

        x1 = int((region[0] + region[2]) )
        y1 = int((region[1] + region[3]) )

        cv2.rectangle(unscaled_img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        plt.imshow("image", unscaled_img)
        plt.show(
    

if __name__ == "__main__":
    main()
