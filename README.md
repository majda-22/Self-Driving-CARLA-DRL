# Self-Driving-CARLA-DRL
building and training a car in the CARLA simulator using Deep Reinforcement Learning


In this project we aim to built a self driving car using deep learning algorithms and CARLA simulator where we are going to test and train our agent.
 We have subdivised the project into smaller parts

# 1-Implementation of the sensors (camera and radar) to gather information about the surrounding objects 
           1-The cameratest.py is used to attach the camera to the agent and gather data(images) and save it locally 
           2-The radartest.ipynb is used to attach the radar to the agent and get information (velocity , depth , azimutude, altidtude) of the surrounding object
# 2-Integration of YOLOv8 to the agent to detect and classify the Object 
            1- the trainyolo.ipnyb to train the model to detect traffic lights(red,green,yellow)
           2- The yolotest.py is used to classify the surrounding object to classes(traffic_lights,cars,people...)
           3- the controlyolo.py is used to control the speed based on the results of YOLOv8 detection
# 3-The Deep Q Network used to control the agent 
           1- The testing.py is used to test the effecient of the model
           2- The training.py is used to train the model
           3- The DQNetwork.py is the architectur of the CNN used to detect the Actions based on the state and Qvalues of (state,action)
           4- The dqncarla.py is used to setup the carla environement and to apply the model and drive
