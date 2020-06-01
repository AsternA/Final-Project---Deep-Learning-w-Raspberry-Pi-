# Final-Project---Deep-Learning-w-Raspberry-Pi-
Final Engineering Project - Electrical Engineering- Afeka academic college of Engineering
1. Background
Our Project: Within our project there are two major aspects. One task is the building of an autonomous multirotor drone complete with a flight controller - Pixhawk. The second task is incorporating a mini (companion) computer along with the flight controller thus being able to identify an object (or number) being presented to the camera by the user. Once an object has been identified the drone will begin its mission without any user input or interference. Each object will be correlated to a specific mission. 

The companion computer will be a Raspberry Pi 3B model. A USB powered camera will be connected to the RPi. Once an object is presented to the camera the RPi will give a complete mission for the Pixhawk to perform. 

My solution was to retrain a Tiny YOLO V3 model to my custom objects. Because the RPi has limited resources we chose the TINY model over the regular.

1.1 Attached Files

  a. Yolo.py
  The main file in the project. Init the yolo model and the camera. The camera scans for our objects within the frames and begins to count the number of consecutive frames the object appears. Once we have reached our Threshold a mission is loaded from the file mission_import.py
  
  b. mission_cancel.py
  Should we show an object to the camera and regret before taking off. The mission Led will begin to blink, giving the user a chance to show the "cancel" object to the camera to abort the mission
  
  c. mission_import.py
  Here the proper mission will be loaded according to the object that was recognized. Missions are in .txt format as they have been planned using the Mission Planner program, and uploaded to the RPi prior to powering on. The file consists of helper functions designed to translate the .txt file to pixhawk commands.
  
  d. mission_x.txt
  Missions that have been planned using the Mission Planner program and exported as .txt files
  
  e. image_gen.py
  Generates images from video. To create a dataset I filmed my objects from all different angles, then used this file to convert the video files into images
  
  f. YOLO folder
  
    i.    obj.names                      - names of our objects
    ii.   Yolov3-tiny-obj.cfg            - tiny yolo config file
    iii.  Yolov3-tiny-obj_final.weights  - yolo weights
    
2. Training

  2.1 My Machine
  
  OS:         Ubuntu 16.04 LTS
  
  Processor:  intel i5 760@2.8 GHz x4
  
  RAM:        16GB DDR3
  
  GPU:        Nvidia Geforce GTX 1080
  
  2.2 Order of Operations 
  
    a. Darknet install from https://github.com/AlexeyAB/darknet
    b. film my objects from various angles and create images
    c. mark my objects using Yolo_Mark https://github.com/AlexeyAB/Yolo_mark
    d. train
  
3. Implementation

  Language: Python 3 
  
  3.1 Libraries Needed:
  
    a. For Training: 
                 Recommended: 	               My Machine:
                 CMake  >= 3.12                    3.16.5
                 CUDA      10.0     	       10.0
                 OpenCV >= 2.4  	               3.4
                 cuDNN  >= 7.0	               7.6
      GPU with Compute Capability >= 3.0           6.1
      
   b. For Interence (RPi):
   
      OpenCV - 4.2 (Installed from source!)
      Dronekit
      Dronekit – sitl (For Simulation Only)
      Numpy
      Imutils
      
 3.2 How it works
 
  a. Running main file: Yolo.py
  
    - The model is loaded into the system and the camera starts rolling
    - each frame is loaded into the model to look for our objects. If an object is not found we move to the next frame resetting all   
    counters, if an object is found the object counter advances by one and we continue to the next frame.
    - First we must show the the "arm" object. Only after the arm led has been lit, then we can show the mission object. 
    - If the threshold amount of frames has been achived by a counter we forward the mission id of that object to the missions files.
    - The mission led will begin blink, giving us a chance to "abort" the mission by showing the abort object.
    - using the Dronekit Library, we connect to the pixhawk. The RPi then sends commands (waypoints) to the pixhawk and the pixhawk 
    sends back acknowledgments and when the waypoint has been reached. Once a waypoint has been reached the next waypoint is sent.
    - Once the Copter has landed and declared "DISARMED", the RPi will disconnect from the pixhawk and activate the camera and start to 
    look for our objects
    
4. Important Notes
    
    - Mission 1: Fly to coordinates, rasie altitude to 100m wait 30 seconds, return to land
    - Mission 2: Fly to coordinates, perform circle with radius 50 m, return to land
    - Mission 3: Fly to coordinates, land to drop off equipment, takeoff, return to land
    - Mission 4: Fly to coordinates, drop item from 50 m altitude, return to land
    - All missions were checked with mavproxy simulator to verify they were performed to completetion
    - Running Tiny Yolov3 on RPi "as is" is exteremly slow! ~ 0.3 FPS
    - changing the following line in yolo.py improves the FPS to ~ 3 FPS!!! With out a noticeable change in predict accuracy!
    
    OLD:
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)

    NEW:

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (128, 128),
        swapRB=True, crop=False)
        
5. Running the Simulator
  
  a. Libraries (install using pip within environment):
  
    - Dronekit-sitl
    - MavProxy
  
  b. Equipment I used:
  
    - MacBook
    - Windows PC
    - RPi
  
  c. Instructions:
  
    - On macbook, open terminal and enter the following command:  (Coordinantes of Home Position in Park Ha Yarkon)
    
    dronekit-sitl copter-3.3 --home=32.103605,34.814987,0.0
    - On second terminal window enter the following command:      (Out1 – IP address of RPi , Out2 – IP address of windows PC)
    
    mavproxy.py --master=tcp:127.0.0.1:5760
     --out=udp:10.100.102.24:14550 --out=udp:10.100.102.29:14550
    
    - On windows PC, Open Mission Planner and connect with UDP (default 14550 port)
    - On RPi in mission_import.py file, change the connect default value to YOUR RPi IP address
    - run main yolo.py file

Video of the Project:
https://www.youtube.com/watch?v=LQ6SFYkovR0&t=2s
    



    

      

  
