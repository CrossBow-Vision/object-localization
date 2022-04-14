import csv
import math

import os
import glob
from pathlib import Path

from PIL import Image
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, Reshape , Rescaling
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon

# 0.35, 0.5, 0.75, 1.0
ALPHA = 1.0

# 96, 128, 160, 192, 224
IMAGE_SIZE = 224

EPOCHS = 200
BATCH_SIZE = 11
PATIENCE = 50

MULTI_PROCESSING = False
THREADS = 1

ANNOT_DIR = "annotations/"
# VALIDATION_CSV = "validation.csv"


class DataGenerator(Sequence):
    
    def __init__(self, annotations_dir):              # annotations_dir :  "annotations/"
        
        annotations_path = sorted(list(Path(annotations_dir) .glob('**/*.json')))        
        self.paths = []        
        self.coords = np.zeros( len(annotations_path), 4))      # numpy array of no. of rows * 4 
    
        for index, json_file in enumerate(annotations_path):

            f = open(json_file)
            data = json.load(f)
            
            self.coords[index, 0] = data['shapes'][0]["points"][0][0]
            self.coords[index, 1] = data['shapes'][0]["points"][0][1]
            self.coords[index, 2] = data['shapes'][0]["points"][1][0] - data['shapes'][0]["points"][0][0]
            self.coords[index, 3] = data['shapes'][0]["points"][1][1] - data['shapes'][0]["points"][0][1]

            self.paths.append(data["imagePath"][3:])

            f.close()



    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = cv2.imread(f)
            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))               

        return batch_images, batch_coords
    
    
 

class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)
            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)

            diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
            diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:,2] * gt[:,3]
            area_pred = pred[:,2] * pred[:,3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        print(" - val_iou: {} - val_mse: {}".format(iou, mse))

def create_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA)

    # to freeze layers
    for layer in model.layers:
        layer.trainable = trainable

    x = model.layers[-1].output
    x = Conv2D(1, kernel_size=3, name="coords")(x)
    x = MaxPooling2D()(x)
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)

def main():
    model = create_model()
    model.summary()

    train_datagen = DataGenerator(ANNOT_DIR)
    validation_datagen = Validation(generator=DataGenerator(ANNOT_DIR))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[])

    checkpoint = ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max")
    
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")

    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=10, min_lr=1e-4, verbose=1, mode="max")

    model.fit(train_datagen, epochs=EPOCHS, callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                        workers=THREADS, use_multiprocessing=MULTI_PROCESSING, shuffle=False, verbose=1)



if __name__ == "__main__":
    main()
