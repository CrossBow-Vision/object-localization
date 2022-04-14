import glob
import cv2
import numpy as np

from train import create_model, IMAGE_SIZE
from keras.applications.mobilenetv2 import preprocess_input

WEIGHTS_FILE = "model-0.52.h5"
IMAGES = "images_resized/*png"

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for img in glob.glob(IMAGES):

        region = model.predict(x=np.array([img]))[0]

        x0 = int(region[0] )
        y0 = int(region[1] )

        x1 = int((region[0] + region[2]) )
        y1 = int((region[1] + region[3]) )

        cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
