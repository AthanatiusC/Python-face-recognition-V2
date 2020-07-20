from library import FacialRecognition
import argparse,cv2,os

if __name__ == "__main__":
    try:
        FaceRecog = FacialRecognition()
        FaceRecog.Start()  
    except (KeyboardInterrupt, SystemExit):
        os._exit(0)