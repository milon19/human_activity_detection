# human_activity_detection


## 1. Install Dependency
```
pip install -r requirement.txt
```

## 2. Download require pre-trained models

1. resnet-34_kinetics: <a href="https://drive.google.com/file/d/19HkKU2dT9BZtOm-0nNCJ1ht8WzK5Nma8/view?usp=sharing">Click Here.</a>
2. Emotion Detection Model: <a href="https://drive.google.com/file/d/1_EWcxfWPuUmc9kTLgLn6ZGLg0Da0NRWl/view?usp=sharing">Click here</a>

## 3. Files structure
        +--action_recognition_kinetics.txt 
        +--example_activities.mp4 
        +--haarcascade_frontalface_default.xml 
        +--human_activity_reco.py 
        +--human_activity_reco_deque.py 
        +--model.h5 
        +--requirment.txt 
        +--resnet-34_kinetics.onnx 
      
 ## Run
 Open cmd and type <br>
 For video input:
  ```
  python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
  ```
 For camera(live) input:
  ```
  python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt
  ```
