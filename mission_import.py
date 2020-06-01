#################################################
#                                               #
# Written by: Almog Stern                       #
# Date: 15.4.20                                 #
# Missions to be given to the Pixhawk           #
# (With help from Dronekit Examples)            #
#                                               #   
#################################################
# Library Imports
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
from pymavlink import mavutil
import argparse
import time
import random
import math
# Connection Handle
parser = argparse.ArgumentParser()
#parser.add_argument('--connect', default='/dev/ttyS0')
parser.add_argument('--connect', default="udp:10.100.102.31:14550")
args = parser.parse_args()


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5



def distance_to_current_waypoint(vehicle):
    """
    Gets distance in metres to the current waypoint. 
    It returns None for the first waypoint (Home location).
    """
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None
    missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    distancetopoint = get_distance_metres(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint

def readmission(vehicle, aFileName):
    """
    Load a mission from a file into a list. The mission definition is in the Waypoint file
    format (http://qgroundcontrol.org/mavlink/waypoint_protocol#waypoint_file_format).

    This function is used by upload_mission().
    """
    print("\nReading mission from file: %s" % aFileName)
    cmds = vehicle.commands
    missionlist=[]
    with open(aFileName) as f:
        for i, line in enumerate(f):
            if i==0:
                if not line.startswith('QGC WPL 110'):
                    raise Exception('File is not supported WP version')
            else:
                linearray=line.split('\t')
                ln_index=int(linearray[0])
                ln_currentwp=int(linearray[1])
                ln_frame=int(linearray[2])
                ln_command=int(linearray[3])
                ln_param1=float(linearray[4])
                ln_param2=float(linearray[5])
                ln_param3=float(linearray[6])
                ln_param4=float(linearray[7])
                ln_param5=float(linearray[8])
                ln_param6=float(linearray[9])
                ln_param7=float(linearray[10])
                ln_autocontinue=int(linearray[11].strip())
                cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
                missionlist.append(cmd)
    return missionlist


def upload_mission(vehicle, aFileName):
    """
    Upload a mission from a file. 
    """
    #Read mission from file
    missionlist = readmission(vehicle, aFileName)
    
    print("\nUpload mission from a file: %s" % aFileName)
    #Clear existing mission from vehicle
    print(' Clear mission')
    cmds = vehicle.commands
    cmds.clear()
    #Add new mission to vehicle
    for command in missionlist:
        cmds.add(command)
    print(' Upload mission')
    vehicle.commands.upload()

            
def arm_and_takeoff(vehicle, aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...")
        time.sleep(1)

        
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print("Waiting for arming...")
        time.sleep(1)

    print("[INFO] Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print("[INFO] Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("[INFO] Reached target altitude")
            break
        time.sleep(1)
            
def which_mission(mid):
    if mid == 1:
        do_mission('mission_1.txt')
    elif mid == 2:
        do_mission('mission_2.txt')
    elif mid == 3:
        do_mission('mission_3.txt')
    elif mid == 4:
        do_mission('mission_4.txt')
    else:
        print('[ERROR] No mission matches number...')
    
    return 0

def do_mission(mission_id):
    import_mission_filename = mission_id
    print('[INFO] Connecting to vehicle on: %s' % args.connect)
    vehicle = connect(args.connect, baud=57600, wait_ready=True)
    # From Copter 3.3 you will be able to take off using a mission item. Plane must take off using a mission item (currently).
    arm_and_takeoff(vehicle, 25)
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...")
        time.sleep(1)
    #Upload mission from file
    missionslist = upload_mission(vehicle, import_mission_filename)
    print("Starting mission")
    # Reset mission set to first (0) waypoint
    vehicle.commands.next=0

    # Set mode to AUTO to start mission
    vehicle.mode = VehicleMode("AUTO")


    # Monitor mission. 
    # Demonstrates getting and setting the command number 
    # Uses distance_to_current_waypoint(), a convenience function for finding the 
    #   distance to the next waypoint.
    while True:
        nextwaypoint=vehicle.commands.next
        print('Distance to waypoint (%s): %s' % (nextwaypoint, distance_to_current_waypoint(vehicle)))
      
        if vehicle.armed == False:
            break
        time.sleep(5)

    #Close vehicle object before exiting script
    print("Close vehicle object")
    vehicle.close()