import os
import cv2
import tensorflow as tf
import numpy as np


class AugmentImages():
    def __init__(self):
        self.path = "../trainingData"


    # =================================
    def augmentData(self):
        # --- read images ---
        count = 0
        data = os.listdir(self.path)

        for x in range(len(data)):
            folder = os.listdir(self.path + "/" + str(count))
            for file in folder:
                # --- read image + save augmented image ---
                srcImage = cv2.imread(self.path + "/" + str(count) + "/" + file)
                augmentedImageRotation1 = tf.image.rot90(srcImage, k=1)
                augmentedImageRotation2 = tf.image.rot90(srcImage, k=2)
                augmentedImageRotation3 = tf.image.rot90(srcImage, k=3)
                # augmentedImageLeftRight = tf.image.random_flip_left_right(srcImage) makes no sense with left/right turn sign
                augmentedImageUpDown = tf.image.random_flip_up_down(srcImage)
                augmentedImageHue = tf.image.random_hue(srcImage, 0.08)
                augmentedImageSaturation = tf.image.random_saturation(srcImage, 0.6, 1.6)
                augmentedImageBrightness = tf.image.random_brightness(srcImage, 0.05)
                augmentedImageContrast = tf.image.random_contrast(srcImage, 0.7, 1.3)

                cv2.imwrite(self.path + "/" + str(count) + "/" + "augRotation1_" + file, np.float32(augmentedImageRotation1))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augRotation2_" + file, np.float32(augmentedImageRotation2))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augRotation3_" + file, np.float32(augmentedImageRotation3))
                # cv2.imwrite(self.path + "/" + str(count) + "/" + "augLeftRight_" + file, np.float32(augmentedImageLeftRight))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augUpDown_" + file, np.float32(augmentedImageUpDown))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augHue_" + file, np.float32(augmentedImageHue))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augSaturation_" + file, np.float32(augmentedImageSaturation))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augBrightness_" + file, np.float32(augmentedImageBrightness))
                cv2.imwrite(self.path + "/" + str(count) + "/" + "augContrast_" + file, np.float32(augmentedImageContrast))
            print("Successfully created augmented images for folder " + str(count) + "/42.")
            count += 1


    # =================================
    def delAugmentedImages(self):
        # --- read images ---
        print("initializing...")
        count = 0
        num = 0
        data = os.listdir(self.path)

        for x in range(len(data)):
            folder = os.listdir(self.path + "/" + str(count))
            for file in folder:
                srcImage = self.path + "/" + str(count) + "/" + file
                if "aug" in srcImage:
                    os.remove(srcImage)
                    num += 1
            count += 1
        print("All " + str(num) + " augmented images deleted.")
