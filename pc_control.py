import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER, windll
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import time
import os
import vlc

# Parameters
width, height = 1280, 720
gestureThreshold = 300

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Create a named window and set it to stay on top
cv2.namedWindow("Image")
hwnd = windll.user32.FindWindowW(None, "Image")
windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
mode = None  # None, 'volume', 'brightness'
music_playing = False
music_player = None

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Variables for detecting fists and specific gesture
fist_start_time = None
gesture_start_time = None
fist_duration = 3  # seconds
gesture_duration = 3  # seconds

def change_volume(distance):
    vol = np.interp(distance, [50, 300], [minVol, maxVol])
    volume.SetMasterVolumeLevel(vol, None)

def change_brightness(distance):
    brightness = np.interp(distance, [50, 300], [0, 100])
    sbc.set_brightness(int(brightness))

def open_folder_and_play_music():
    global music_playing, music_player
    folder_path = "C:\\Users\\Amine\\Contacts\\Downloads\\Music"
    music_file = "salahif.mp3"
    os.startfile(folder_path)
    music_path = os.path.join(folder_path, music_file)
    
    # Initialize VLC player
    if not music_player:
        vlc_instance = vlc.Instance()
        music_player = vlc_instance.media_player_new()
        media = vlc_instance.media_new(music_path)
        music_player.set_media(media)
    
    music_player.play()
    music_playing = True

def toggle_music_play_pause():
    global music_playing, music_player
    if music_player and music_player.is_playing():
        music_player.pause()
        music_playing = False
    else:
        music_player.play()
        music_playing = True

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hands and their landmarks
    hands, img = detector.findHands(img, flipType=False) 

    if len(hands) == 1:
        hand = hands[0]
        if hand["type"] == "Left":
            leftHand = hand
            rightHand = None
        else:
            rightHand = hand
            leftHand = None

    elif len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        # Identify which hand is left and which is right
        if hand1["type"] == "Left":
            leftHand = hand1
            rightHand = hand2
        else:
            leftHand = hand2
            rightHand = hand1
    else:
        leftHand = None
        rightHand = None

    # Check gestures for left hand
    if leftHand:
        lmListLeft = leftHand["lmList"]
        fingersLeft = detector.fingersUp(leftHand)

        if fingersLeft == [0, 1, 0, 0, 0] or fingersLeft == [1, 1, 0, 0, 0]:  # Left hand index finger up
            mode = 'volume'
            cv2.putText(img, "Mode: Volume", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif fingersLeft == [0, 1, 1, 0, 0] or fingersLeft == [1, 1, 1, 0, 0]:  # Left hand index and middle fingers up
            mode = 'brightness'
            cv2.putText(img, "Mode: Brightness", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            mode = None

    # Perform actions based on the right hand gestures and mode
    if rightHand:
        lmListRight = rightHand["lmList"]
        fingersRight = detector.fingersUp(rightHand)

        if mode in ['volume', 'brightness']:
            # Calculate the distance between thumb and index finger
            x1, y1 = lmListRight[4][0], lmListRight[4][1]
            x2, y2 = lmListRight[8][0], lmListRight[8][1]
            distance = np.hypot(x2 - x1, y2 - y1)

            # Draw the line between thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), cv2.FILLED)

            if mode == 'volume':
                change_volume(distance)
            elif mode == 'brightness':
                change_brightness(distance)

    # Check if both hands form fists
    if leftHand and rightHand:
        if detector.fingersUp(leftHand) == [0, 0, 0, 0, 0] and detector.fingersUp(rightHand) == [0, 0, 0, 0, 0]:
            if fist_start_time is None:
                fist_start_time = time.time()
            else:
                elapsed_time = time.time() - fist_start_time
                if elapsed_time > fist_duration:
                    break
        else:
            fist_start_time = None

        # Check for specific gesture to open folder and play music
        if detector.fingersUp(leftHand) == [1, 1, 0, 0, 1] and detector.fingersUp(rightHand) == [1, 1, 0, 0, 1]:
            if gesture_start_time is None:
                gesture_start_time = time.time()
            else:
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time > gesture_duration and not music_playing:
                    open_folder_and_play_music()
                    gesture_start_time = None  # Reset to avoid repeated launches
        else:
            gesture_start_time = None

        # Check for gesture to play/pause music
        if music_playing and detector.fingersUp(leftHand) == [1, 1, 1, 1, 1] and detector.fingersUp(rightHand) == [1, 1, 1, 1, 1]:
            toggle_music_play_pause()
    else:
        fist_start_time = None
        gesture_start_time = None

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import time
import os
import subprocess
import psutil

# Parameters
width, height = 1280, 720
gestureThreshold = 300

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
mode = None  # None, 'volume', 'brightness'
music_playing = False
music_process = None

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Variables for detecting fists and specific gesture
fist_start_time = None
gesture_start_time = None
fist_duration = 3  # seconds
gesture_duration = 3  # seconds

def change_volume(distance):
    vol = np.interp(distance, [50, 300], [minVol, maxVol])
    volume.SetMasterVolumeLevel(vol, None)

def change_brightness(distance):
    brightness = np.interp(distance, [50, 300], [0, 100])
    sbc.set_brightness(int(brightness))

def open_folder_and_play_music():
    global music_playing, music_process
    folder_path = "C:\\Users\\Amine\\Contacts\\Downloads\\Music"
    music_file = "salahif.mp3"
    os.startfile(folder_path)
    music_process = subprocess.Popen(["vlc", os.path.join(folder_path, music_file)], shell=True)
    music_playing = True

def toggle_music_play_pause():
    global music_playing, music_process
    if music_process and music_process.poll() is None:  # Check if the process is still running
        music_process.terminate()
        music_playing = False
    else:
        open_folder_and_play_music()

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hands and their landmarks
    hands, img = detector.findHands(img, flipType=False) 

    if len(hands) == 1:
        hand = hands[0]
        if hand["type"] == "Left":
            leftHand = hand
            rightHand = None
        else:
            rightHand = hand
            leftHand = None

    elif len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        # Identify which hand is left and which is right
        if hand1["type"] == "Left":
            leftHand = hand1
            rightHand = hand2
        else:
            leftHand = hand2
            rightHand = hand1
    else:
        leftHand = None
        rightHand = None

    # Check gestures for left hand
    if leftHand:
        lmListLeft = leftHand["lmList"]
        fingersLeft = detector.fingersUp(leftHand)

        if fingersLeft == [0, 1, 0, 0, 0] or fingersLeft == [1, 1, 0, 0, 0]:  # Left hand index finger up
            mode = 'volume'
            cv2.putText(img, "Mode: Volume", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif fingersLeft == [0, 1, 1, 0, 0] or fingersLeft == [1, 1, 1, 0, 0]:  # Left hand index and middle fingers up
            mode = 'brightness'
            cv2.putText(img, "Mode: Brightness", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            mode = None

    # Perform actions based on the right hand gestures and mode
    if rightHand:
        lmListRight = rightHand["lmList"]
        fingersRight = detector.fingersUp(rightHand)

        if mode in ['volume', 'brightness']:
            # Calculate the distance between thumb and index finger
            x1, y1 = lmListRight[4][0], lmListRight[4][1]
            x2, y2 = lmListRight[8][0], lmListRight[8][1]
            distance = np.hypot(x2 - x1, y2 - y1)

            # Draw the line between thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), cv2.FILLED)

            if mode == 'volume':
                change_volume(distance)
            elif mode == 'brightness':
                change_brightness(distance)

    # Check if both hands form fists
    if leftHand and rightHand:
        if detector.fingersUp(leftHand) == [0, 0, 0, 0, 0] and detector.fingersUp(rightHand) == [0, 0, 0, 0, 0]:
            if fist_start_time is None:
                fist_start_time = time.time()
            else:
                elapsed_time = time.time() - fist_start_time
                if elapsed_time > fist_duration:
                    break
        else:
            fist_start_time = None

        # Check for specific gesture to open folder and play music
        if detector.fingersUp(leftHand) == [1, 1, 0, 0, 1] and detector.fingersUp(rightHand) == [1, 1, 0, 0, 1]:
            if gesture_start_time is None:
                gesture_start_time = time.time()
            else:
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time > gesture_duration and not music_playing:
                    open_folder_and_play_music()
                    gesture_start_time = None  # Reset to avoid repeated launches
        else:
            gesture_start_time = None

        # Check for gesture to play/pause music
        if detector.fingersUp(leftHand) == [1, 1, 1, 1, 1] and detector.fingersUp(rightHand) == [1, 1, 1, 1, 1]:
            toggle_music_play_pause()

    else:
        fist_start_time = None
        gesture_start_time = None

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import time
import os
import subprocess
import psutil

# Parameters
width, height = 1280, 720
gestureThreshold = 300

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
mode = None  # None, 'volume', 'brightness'
music_playing = False
music_process = None

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Variables for detecting fists and specific gesture
fist_start_time = None
gesture_start_time = None
fist_duration = 3  # seconds
gesture_duration = 3  # seconds

def change_volume(distance):
    vol = np.interp(distance, [50, 300], [minVol, maxVol])
    volume.SetMasterVolumeLevel(vol, None)

def change_brightness(distance):
    brightness = np.interp(distance, [50, 300], [0, 100])
    sbc.set_brightness(int(brightness))

def open_folder_and_play_music():
    global music_playing, music_process
    folder_path = "C:\\Users\\Amine\\Contacts\\Downloads\\Music"
    music_file = "salahif.mp3"
    os.startfile(folder_path)
    music_process = subprocess.Popen(["start", os.path.join(folder_path, music_file)], shell=True)
    music_playing = True

def toggle_music_play_pause():
    global music_playing, music_process
    if music_process:
        for proc in psutil.process_iter():
            if "wmplayer.exe" in proc.name() or "vlc.exe" in proc.name():  # Depending on the music player used
                proc.kill()
                music_playing = False
                return
    if not music_playing:
        open_folder_and_play_music()

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hands and their landmarks
    hands, img = detector.findHands(img, flipType=False) 

    if len(hands) == 1:
        hand = hands[0]
        if hand["type"] == "Left":
            leftHand = hand
            rightHand = None
        else:
            rightHand = hand
            leftHand = None

    elif len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        # Identify which hand is left and which is right
        if hand1["type"] == "Left":
            leftHand = hand1
            rightHand = hand2
        else:
            leftHand = hand2
            rightHand = hand1
    else:
        leftHand = None
        rightHand = None

    # Check gestures for left hand
    if leftHand:
        lmListLeft = leftHand["lmList"]
        fingersLeft = detector.fingersUp(leftHand)

        if fingersLeft == [0, 1, 0, 0, 0] or fingersLeft == [1, 1, 0, 0, 0]:  # Left hand index finger up
            mode = 'volume'
            cv2.putText(img, "Mode: Volume", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif fingersLeft == [0, 1, 1, 0, 0] or fingersLeft == [1, 1, 1, 0, 0]:  # Left hand index and middle fingers up
            mode = 'brightness'
            cv2.putText(img, "Mode: Brightness", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            mode = None

    # Perform actions based on the right hand gestures and mode
    if rightHand:
        lmListRight = rightHand["lmList"]
        fingersRight = detector.fingersUp(rightHand)

        if mode in ['volume', 'brightness']:
            # Calculate the distance between thumb and index finger
            x1, y1 = lmListRight[4][0], lmListRight[4][1]
            x2, y2 = lmListRight[8][0], lmListRight[8][1]
            distance = np.hypot(x2 - x1, y2 - y1)

            # Draw the line between thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), cv2.FILLED)

            if mode == 'volume':
                change_volume(distance)
            elif mode == 'brightness':
                change_brightness(distance)

    # Check if both hands form fists
    if leftHand and rightHand:
        if detector.fingersUp(leftHand) == [0, 0, 0, 0, 0] and detector.fingersUp(rightHand) == [0, 0, 0, 0, 0]:
            if fist_start_time is None:
                fist_start_time = time.time()
            else:
                elapsed_time = time.time() - fist_start_time
                if elapsed_time > fist_duration:
                    break
        else:
            fist_start_time = None

        # Check for specific gesture to open folder and play music
        if detector.fingersUp(leftHand) == [1, 1, 0, 0, 1] and detector.fingersUp(rightHand) == [1, 1, 0, 0, 1]:
            if gesture_start_time is None:
                gesture_start_time = time.time()
            else:
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time > gesture_duration and not music_playing:
                    open_folder_and_play_music()
                    gesture_start_time = None  # Reset to avoid repeated launches
        else:
            gesture_start_time = None

        # Check for gesture to play/pause music
        if detector.fingersUp(leftHand) == [1, 1, 1, 1, 1] and detector.fingersUp(rightHand) == [1, 1, 1, 1, 1]:
            toggle_music_play_pause()

    else:
        fist_start_time = None
        gesture_start_time = None

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import time
import os
import subprocess

# Parameters
width, height = 1280, 720
gestureThreshold = 300

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
mode = None  # None, 'volume', 'brightness'

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Variables for detecting fists and specific gesture
fist_start_time = None
gesture_start_time = None
fist_duration = 3  # seconds
gesture_duration = 3  # seconds

def change_volume(distance):
    vol = np.interp(distance, [50, 300], [minVol, maxVol])
    volume.SetMasterVolumeLevel(vol, None)

def change_brightness(distance):
    brightness = np.interp(distance, [50, 300], [0, 100])
    sbc.set_brightness(int(brightness))

def open_folder_and_play_music():
    folder_path = "C:\\Users\\Amine\\Contacts\\Downloads\\Music"
    music_file = "salahif.mp3"
    os.startfile(folder_path)
    subprocess.Popen(["start", os.path.join(folder_path, music_file)], shell=True)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hands and their landmarks
    hands, img = detector.findHands(img, flipType=False) 

    if len(hands) == 1:
        hand = hands[0]
        if hand["type"] == "Left":
            leftHand = hand
            rightHand = None
        else:
            rightHand = hand
            leftHand = None

    elif len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        # Identify which hand is left and which is right
        if hand1["type"] == "Left":
            leftHand = hand1
            rightHand = hand2
        else:
            leftHand = hand2
            rightHand = hand1
    else:
        leftHand = None
        rightHand = None

    # Check gestures for left hand
    if leftHand:
        lmListLeft = leftHand["lmList"]
        fingersLeft = detector.fingersUp(leftHand)

        if fingersLeft == [0, 1, 0, 0, 0] or fingersLeft == [1, 1, 0, 0, 0]:  # Left hand index finger up
            mode = 'volume'
            cv2.putText(img, "Mode: Volume", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif fingersLeft == [0, 1, 1, 0, 0] or fingersLeft == [1, 1, 1, 0, 0]:  # Left hand index and middle fingers up
            mode = 'brightness'
            cv2.putText(img, "Mode: Brightness", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            mode = None

    # Perform actions based on the right hand gestures and mode
    if rightHand:
        lmListRight = rightHand["lmList"]
        fingersRight = detector.fingersUp(rightHand)

        if mode in ['volume', 'brightness']:
            # Calculate the distance between thumb and index finger
            x1, y1 = lmListRight[4][0], lmListRight[4][1]
            x2, y2 = lmListRight[8][0], lmListRight[8][1]
            distance = np.hypot(x2 - x1, y2 - y1)

            # Draw the line between thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), cv2.FILLED)

            if mode == 'volume':
                change_volume(distance)
            elif mode == 'brightness':
                change_brightness(distance)

    # Check if both hands form fists
    if leftHand and rightHand:
        if detector.fingersUp(leftHand) == [0, 0, 0, 0, 0] and detector.fingersUp(rightHand) == [0, 0, 0, 0, 0]:
            if fist_start_time is None:
                fist_start_time = time.time()
            else:
                elapsed_time = time.time() - fist_start_time
                if elapsed_time > fist_duration:
                    break
        else:
            fist_start_time = None

        # Check for specific gesture to open folder and play music
        if detector.fingersUp(leftHand) == [1, 1, 0, 0, 1] and detector.fingersUp(rightHand) == [1, 1, 0, 0, 1]:
            if gesture_start_time is None:
                gesture_start_time = time.time()
            else:
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time > gesture_duration:
                    open_folder_and_play_music()
                    gesture_start_time = None  # Reset to avoid repeated launches
        else:
            gesture_start_time = None

    else:
        fist_start_time = None
        gesture_start_time = None

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""