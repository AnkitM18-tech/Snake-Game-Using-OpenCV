import math,cvzone,random
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Setting the Video
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Hand Detection
detector = HandDetector(detectionCon=0.8,maxHands=1)

# Creating game class
class SnakeGame:
    def __init__(self,pathFood):
        self.points = [] #all points of the snake
        self.lengths = [] # distance between each point
        self.currentLength = 0 # total length of the snake
        self.allowedLength = 150 # total allowed length of snake
        self.previousHead = 0, 0 # previous head point

        # Food
        self.imgFood = cv2.imread(pathFood,cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0,0
        self.randomFoodLocation()

        # Game
        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100,1000),random.randint(100,600) 

    def update(self,imgMain, currentHead):
        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over",(200,400),scale=7,thickness=5,offset=20)
            cvzone.putTextRect(imgMain, f"Your Score : {self.score}",(200,550), scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead
            self.points.append([cx, cy])
            distance = math.hypot(cx-px,cy-py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    # check the length continuosly
                    if self.currentLength < self.allowedLength:
                        break

            # Check if the snake ate the food
            rx, ry = self.foodPoint
            if rx - self.wFood//2 < cx < rx + self.wFood//2 and ry - self.hFood//2 < cy < ry + self.hFood//2:
                # print("Ate")
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 10
                print(self.score)

            # Draw Snake
            if self.points:
                for i,point in enumerate(self.points):
                    if i!= 0:
                        cv2.line(imgMain,self.points[i-1],self.points[i],(0,0,255),20)
                cv2.circle(imgMain,self.points[-1],20,(200,0,200),cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain,self.imgFood,(rx-self.wFood//2, ry-self.hFood//2))

            cvzone.putTextRect(imgMain, f"Score : {self.score}",(50,80), scale=3, thickness=3, offset=10)

            # Check for collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(imgMain,[pts],False,(0,200,0),3)
            minDist = cv2.pointPolygonTest(pts,(cx,cy),True)
            # print(minDist)

            if -1 <= minDist <= 1:
                # print("Hit")
                self.gameOver = True
                self.points = [] #all points of the snake
                self.lengths = [] # distance between each point
                self.currentLength = 0 # total length of the snake
                self.allowedLength = 150 # total allowed length of snake
                self.previousHead = 0, 0 # previous head point
                self.randomFoodLocation()

        return imgMain

game = SnakeGame("./Donut.png")

# Showing the captured Video
while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img,flipType=False)

    # Finding landmark points and getting index finger lm point 8
    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img,pointIndex)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    # Reseting the game
    if key == ord("r"):
        game.score = 0
        game.gameOver = False
    if key == ord("q"):
        break