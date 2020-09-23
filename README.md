# CrickNet: Computer Vision based cricket practice sessions analysis

This is repository contains [output_videos](output), [testing_codes](CrickNet_codes), and [training_files](CrickNet_object_detection_training_files)

---
## CrickNet Outputs
Run [CrickNet]CrickNet_video.py) to see the overall analytical results such as ball speed, impact point, swing speed, and performance.

Example of impact point of balll & bat:
![Impact Point](https://github.com/ashish-AIML/Cricket_Analytics_Computer_Vision/blob/master/output/impact_point.JPG "Ball-Bat Impact Point") 

For a better bowling analysis, background is segmented as white & only objects & their motion are visualized. [White_bakground_analysis](white_bakground.mp4)

Bowling Angle is calculated based on the x-y coordinates. [Bowling_angle](cricknet_angle_video.mp4)
***
---
## Training Files
YOLOv3 object detection model is used to train the ball, bat, pitch & batsman. [obj.cfg](obj.cfg) contains yolo configuration. [obj_new.data](obj_new.data) contains training files, class numbers and backup folder path. [obj_new.names](obj_new.names) contains the class labels, here it is ball, bat, pitch, and batsman.
***
## Maths logics
Ball motion time & speed is calculated based on the frame-rate (in our case it is 120 FPS)
```
time = hit_frameA/120

speed= 22/time
```

The impact point of ball & bat is calculated based on the distance between the ball, bat, pitch
```
dist=math.sqrt(((bcx-cx)*(bcx-cx))+((bcy-cy)*(bcy-cy)))
```
The ball impact point on the ground (pitch) is measured in similar to the bat & ball impact point, based on the centroid distance between the two
***

---
## License & Copyright

@ Ashish & team

***
