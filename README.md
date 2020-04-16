# human_activity_detection


## 1. Install Dependency
```
pip install -r requirment.txt
```

## 2. Download require pre-trained models

1. resnet-34_kinetics: <a href="https://drive.google.com/file/d/19HkKU2dT9BZtOm-0nNCJ1ht8WzK5Nma8/view?usp=sharing">Click Here.</a>
2. Emotion Detection Model: <a href="https://drive.google.com/file/d/1_EWcxfWPuUmc9kTLgLn6ZGLg0Da0NRWl/view?usp=sharing">Click here</a>

## 3. Files structure
        +--action_recognition_kinetics.txt <br>
        +--example_activities.mp4 <br>
        +--haarcascade_frontalface_default.xml <br>
        +--human_activity_reco.py <br>
        +--human_activity_reco_deque.py <br>
        +--model.h5 <br>
        +--requirment.txt <br>
        +--resnet-34_kinetics.onnx <br>
      
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
