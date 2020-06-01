
import cv2

cap = cv2.VideoCapture('/home/almogs/workspace/Gen_imgs_from_videos/Videos/IMG_2070.MOV')

if not cap.isOpened():
    print("[ERROR] can not open video file")

i = 0
while cap.isOpened():

    grabbed, frame = cap.read()

    if not grabbed:
        print("[INFO] could not grab frame...")
        break

    cv2.imwrite('img_6_' + str(i) + '.jpg', frame)
    i+=1

cap.release()
cv2.destroyAllWindows()



