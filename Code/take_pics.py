"""
"A Simple Domain Shifting Network for Generating Low Quality Images" implementation

Step 1: Capturing  pairs  of  corresponding  high and low resolution image: High resolution images are displayed on the smallscreen 640x480, which is captured by the built-in camera of the Cozmo.

Input:
Possible High resolution image paths:   original_pascal_voc_images_15_classes - PASCAL VOC Partial dataset  (Ours Zero Shot Learning)
                                        original_pascal_voc_images            - PASCAL VOC Complete dataset (Ours Unsupervised)
                                        original_imagenet_images              - ImageNet Partial dataset    (Used for validation)
      
Output:      
Low resolution images:  cozmo

"""
import cv2
import numpy
from PIL import Image
import os
import cozmo
from cozmo.util import degrees
import glob
import time

def cozmo_program(robot: cozmo.robot.Robot):
    robot.set_head_angle(degrees(5)).wait_for_completed()
    robot.set_lift_height(1).wait_for_completed()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True
    
    directories = ["..\\Dataset\\original_pascal_voc_images_15_classes\\"] # Specify the input directory containing High resolution images meant to be captured by Cozmo
    os.makedirs("..\\Dataset\\cozmo\\", exist_ok=True)  # Create output directory for saving low resolution images caputured by Cozmo.
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    #cv2.moveWindow("window", 1950, 398)
    #cv2.resizeWindow("window", 640, 480)
    latest_img = None
    while latest_img is None:
        latest_img = robot.world.latest_image

    for directory in directories:
        for file in glob.glob(directory+"\\*"):
            img = Image.open(file)
            print("Image: {}".format(file))
            try:
                cv2.imshow("window",cv2.cvtColor(numpy.asarray(img),cv2.COLOR_BGR2RGB))
                robot.set_head_angle(degrees(5)).wait_for_completed()
                robot.set_lift_height(1).wait_for_completed()
                cv2.waitKey(1000)
                latest_img = robot.world.latest_image
                if latest_img is not None:
                    pilImage = latest_img.raw_image
                    pilImage.save("..\\Dataset\\cozmo\\"+file.split("\\")[-1].split(".jpg")[0]+"_copy.jpg", "JPEG")
            except Exception as e:
                print(e)
            finally:
                img.close()
    cv2.destroyAllWindows()

def cozmo_run():
    cozmo.run_program(cozmo_program, use_viewer=False, force_viewer_on_top=False) 
    cozmo.robot.Robot.drive_off_charger_on_connect = False

if __name__ == '__main__':
    cozmo_run()