#################################################
#                                               #
# Written by: Almog Stern                       #
# Date: 15.4.20                                 #
# Most of the code was written by Adrian from   #
# https://www.pyimagesearch.com, modified by    #
# me for my project                             #
#                                               #   
#                                               #
# Yolo Tiny v3 model was trained by Almog Stern #
# based on directions from                      #
# https://github.com/AlexeyAB/darknet#          #
#                                               #   
#################################################
# Library Imports
import numpy as np
import cv2
import os
import argparse
import time
import mission_import
from imutils.video import FPS



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
ap.add_argument("-y", "--yolo", type=str, default="yolo-num-final-proj",
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
ap.add_argument("-u", "--use-gpu", type=bool, default=1,
    help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# load the class labels into my YOLO modle
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny-obj_final.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-tiny-obj.cfg"])

# load my YOLO object detector trained on my dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# CUDA is irrelevent, we are running on RPi, no GPU
# set CUDA as the preferable backend and target
#print("[INFO] setting preferable backend and target to CUDA...")
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the width and height of the frames in the video file
W = None
H = None

#initialize the total number of frames that *consecutively* contain
#a label along with threshold required 
TOTAL_CONSEC_1 = 0
TOTAL_CONSEC_2 = 0
TOTAL_CONSEC_3 = 0
TOTAL_CONSEC_4 = 0
TOTAL_CONSEC_5 = 0
TOTAL_CONSEC_6 = 0
TOTAL_THRESH = 20  # 20 consecutive frames at 2 FPS is 10 secs!

# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (128, 128),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            label_name = LABELS[classIDs[i]]

            if label_name == 'one':
                TOTAL_CONSEC_1 += 1
                TOTAL_CONSEC_2  = 0
                TOTAL_CONSEC_3  = 0
                TOTAL_CONSEC_4  = 0
                TOTAL_CONSEC_5  = 0
                TOTAL_CONSEC_6  = 0
                if TOTAL_CONSEC_1 >= TOTAL_THRESH:
                    print("[INFO] Mission One")
                    mission_import.which_mission(1)
                    time.sleep(5)
                    TOTAL_CONSEC_1 = 0
            if label_name == 'two':
                TOTAL_CONSEC_1  = 0
                TOTAL_CONSEC_2 += 1
                TOTAL_CONSEC_3  = 0
                TOTAL_CONSEC_4  = 0
                TOTAL_CONSEC_5  = 0
                TOTAL_CONSEC_6  = 0
                if TOTAL_CONSEC_2 >= TOTAL_THRESH:
                    print("[INFO] Mission Two")
                    mission_import.which_mission(2)
                    time.sleep(5)
                    TOTAL_CONSEC_2 = 0
            if label_name == 'three':
                TOTAL_CONSEC_1  = 0
                TOTAL_CONSEC_2  = 0
                TOTAL_CONSEC_3 += 1
                TOTAL_CONSEC_4  = 0
                TOTAL_CONSEC_5  = 0
                TOTAL_CONSEC_6  = 0
                if TOTAL_CONSEC_3 >= TOTAL_THRESH:
                    print("[INFO] Mission Three")
                    mission_import.which_mission(3)
                    time.sleep(5)
                    TOTAL_CONSEC_3 = 0
            if label_name == 'four':
                TOTAL_CONSEC_1  = 0
                TOTAL_CONSEC_2  = 0
                TOTAL_CONSEC_3  = 0
                TOTAL_CONSEC_4 += 1
                TOTAL_CONSEC_5  = 0
                TOTAL_CONSEC_6  = 0
                if TOTAL_CONSEC_4 >= TOTAL_THRESH:
                    print("[INFO] Mission Four")
                    mission_import.which_mission(4)
                    time.sleep(5)
                    TOTAL_CONSEC_4 = 0
            if label_name == 'five':
                TOTAL_CONSEC_1  = 0
                TOTAL_CONSEC_2  = 0
                TOTAL_CONSEC_3  = 0
                TOTAL_CONSEC_4  = 0
                TOTAL_CONSEC_5 += 1
                TOTAL_CONSEC_6  = 0
                if TOTAL_CONSEC_5 >= TOTAL_THRESH:
                    print("[INFO] Mission Five")
                    mission_import.which_mission(5)
                    time.sleep(5)
                    TOTAL_CONSEC_5 = 0
            if label_name == 'six':
                TOTAL_CONSEC_1 += 1
                TOTAL_CONSEC_2  = 0
                TOTAL_CONSEC_3  = 0
                TOTAL_CONSEC_4  = 0
                TOTAL_CONSEC_5  = 0
                TOTAL_CONSEC_6  = 0
                if TOTAL_CONSEC_6 >= TOTAL_THRESH:
                    print("[INFO] Mission Six")
                    mission_import.which_mission(6)
                    time.sleep(5)
                    TOTAL_CONSEC_6 = 0
            else:
                TOTAL_CONSEC_1  = 0
                TOTAL_CONSEC_2  = 0
                TOTAL_CONSEC_3  = 0
                TOTAL_CONSEC_4  = 0
                TOTAL_CONSEC_5  = 0
                TOTAL_CONSEC_6  = 0
    
    
    # check to see if the output frame should be displayed to our
    # screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))