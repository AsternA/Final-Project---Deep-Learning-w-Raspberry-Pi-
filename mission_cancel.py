#################################################
#                                               #
# Written by: Almog Stern                       #
# Date: 21.4.20                                 #
#                                               #
# A second set of image detection to detect     #
# the cancelation of the image. Should i want   #
# to cancel a mission after the quad has        #
# accepted the mission                          # 
#                                               #   
#################################################
# Library Imports
import numpy as np
import cv2
import time
import RPi.GPIO as GPIO





def mission_cancel(vs, net, ln, LABELS, mid):
    toggle = 1
    miss_1 = 17
    miss_2 = 27
    miss_3 = 22
    miss_4 = 23
    start_time = time.time()
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    # initialize the width and height of the frames in the video file
    W = None
    H = None
    
    abort = False
    TOTAL_CONSEC_CANCEL = 0
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
                if confidence > 0.5:
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
            0.3)

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
                    
                if label_name == 'Cancel':
                    TOTAL_CONSEC_CANCEL += 1
                    if TOTAL_CONSEC_CANCEL >= 10: # for safety reasons, canceling a mission requires less frames.
                        print('[INFO] ABORTING MISSION!!')
                        abort = True
                        return abort
                        break
                    
        # Toggle mission led to let the user know which mission is about to launch
        if toggle == 1:
            if mid == 1:
                GPIO.output(miss_1, GPIO.HIGH)
                toggle = 0
            if mid == 2:
                GPIO.output(miss_2, GPIO.HIGH)
                toggle = 0        
            if mid == 3:
                GPIO.output(miss_3, GPIO.HIGH)
                toggle = 0
            if mid == 4:
                GPIO.output(miss_4, GPIO.HIGH)
                toggle = 0
            #if mid == 5:
            #    GPIO.output(miss_1, GPIO.HIGH)
            #    toggle = 0
            #if mid == 6:
            #    GPIO.output(miss_1, GPIO.HIGH)
            #    toggle = 0
        else:
            if mid == 1:
                GPIO.output(miss_1, GPIO.LOW)
                toggle = 1
            if mid == 2:
                GPIO.output(miss_2, GPIO.LOW)
                toggle = 1
            if mid == 3:
                GPIO.output(miss_3, GPIO.LOW)
                toggle = 1
            if mid == 4:
                GPIO.output(miss_4, GPIO.LOW)
                toggle = 1
            #if mid == 5:
            #    GPIO.output(miss_1, GPIO.LOW)
            #    toggle = 1
            #if mid == 6:
            #    GPIO.output(miss_1, GPIO.LOW)
            #    toggle = 1
        
        
        
        end_time = time.time()
        if end_time - start_time >=20:   # 20 seconds to cancel the mission
            abort = False
            print('[INFO] Resuming Mission...')
            return abort
        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        

