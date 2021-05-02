import argparse
import os
import cv2
from tqdm import tqdm

## ARGPARSE SECTION
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Valve Filter Preprocesser Function')
parser.add_argument('--train', type=dir_path, default="./dataset/train/", help="Pass in the train data path")
parser.add_argument('--val', type=dir_path, default="./dataset/val/", help="Pass in the val data path")


args = parser.parse_args()
TRAIN_PATH = args.train
VAL_PATH = args.val


## Pre-processing
files_train = os.listdir(TRAIN_PATH)
filtered_train = [file for file in files_train if "depth" in file]
files_val = os.listdir(VAL_PATH)
filtered_val = [file for file in files_val if "depth" in file]


def generate_roi(filter_path, filtered):
    for filename in tqdm(filtered):
        name = filter_path + filename 
    #     name = target1
        cap = cv2.VideoCapture(name)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        filename = name.replace("depth","roi")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#         print(filename)
        out = cv2.VideoWriter(filename, fourcc, 30, (512, 512))

        while True:
            _, frame = cap.read()
            if frame is None:
                break
            output = frame.copy()
            retval,thresh = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(thresh,kernel,iterations = 1)
            out.write(dilation)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        out.release()

print ("Generating Train Data ROI")
generate_roi(TRAIN_PATH, filtered_train)
print ("Generating Val Data ROI")
generate_roi(VAL_PATH, filtered_val)
print ("ROI Generation Complete!")