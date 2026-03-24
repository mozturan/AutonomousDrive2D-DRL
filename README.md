# Autonomous Driving W/ Deep Reinforcement Learning in Lane Keeping

## **SAC with KINEMATICS INPUT TRAINING RESULTS**
<div align="center" style="width: 600px;">
  <video
    src="https://github.com/rafiqollective/thesis/assets/89272933/30c2c8f4-209d-44f6-98f9-c89bc9620ce3"
    controls
    style="display:block; width:100%; max-width:100%; margin:0 auto;"
  ></video>
</div>

## **DDQN-PER with KINEMATICS INPUT TRAINING RESULTS**
<div align="center" style="width: 600px;">
  <video
    src="https://github.com/rafiqollective/thesis/assets/89272933/87b5acfc-98a9-4f52-90ca-cf9d84edcb64"
    controls
    style="display:block; width:100%; max-width:100%; margin:0 auto;"
  ></video>
</div>

<p align="center">
  <a href="https://www.isarconference.org/_files/ugd/6dc816_494830e0cbba49a8a5511403a2727f8d.pdf"><img src="https://img.shields.io/badge/Paper-ISARC%202023-blue?style=flat-square" alt="Paper"/></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Simulator-Donkey%20Car-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square"/>
</p>

---

> **Comparative Analysis of SAC and DDQN in 2D Autonomous Vehicle Simulation: Insights From Grayscale Bird View And Kinematics Inputs**  
> Mehmet Yaşar Osman Özturan
> *Computer Engineering Department, Bursa Uludağ University, Türkiye*  
> **1. International Black Sea Scientific Research and Innovation Congress 2023**

---
## Overview

A comprehensive comparative analysis of two formidable deep reinforcement learning algorithms: **Soft Actor-Critic (SAC)** and **Double Deep Q-Network with Prioritized Experience Replay (DDQN with PER)**. Our primary goal was to discern how the choice of observation space influences the performance of these algorithms.

Our primary goal was to discern how the choice of observation space influences the performance of these algorithms and to offer an alternative to end-to-end deep learning studies carried out with raw sensor data and to show that processed data is much more successful in terms of reinforcement learning algorithms in the autonomous driving system, compared to raw data.

## Method

### Simulation Environment
![Screenshot from 2024-02-14 19-03-17](https://github.com/rafiqollective/thesis/assets/89272933/10953242-8b1b-415b-862d-fda8bbbfb8d4)

- Using Highway-Env simulation.
- The simulated environment was designed to mimic a racetrack scenario.
- Vehicle tasked with lane-keeping and maintaining target speed on a racetrack.
- Testing two different deep reinforcement learning algorithms (SAC and DDQN-PER) with two different observation types (Kinematics and Birdview Images)

### **Action Space :**
Both steering and throttle can be controlled. In fact, *"one_act"* file contains code for the situation where agents control steering only, and *"two_acts"* file contains code for the situation where agents control both steering and throttle. This doc focused on "*two_acts*". 

Action spaces are continuous between **[-1,1]** values. Continuous action space is supported in SAC. For DDQN-PER, we discretize action space to 55 different action.  

### **Observation Spaces**
Two different observation types are testes:
  1. Kinematics
     
![Screenshot from 2024-02-14 19-17-12](https://github.com/rafiqollective/thesis/assets/89272933/351696b0-f9bc-4191-a4cd-c03673fafb3e)

  3. Birdview Images
     
![Screenshot from 2024-02-14 19-18-06](https://github.com/rafiqollective/thesis/assets/89272933/61d7f780-25f8-49e8-b055-bc3cd4379ff6)

### **Reward Funtion**
Designed to Promote:
  - On-road behavior
  - Distance to lane centering
  - Target speed maintenance

![Screenshot from 2024-02-14 19-22-00](https://github.com/rafiqollective/thesis/assets/89272933/779d131e-acb4-4783-a316-c8cda0826848)

** **FOR TARGET SPEED MAINTENANCE WE USE GAUSSIAN FUNCTION**

Terminal conditions:
  - Agent is off road
  - Agent reaches maximum number of steps
  - Agent reaches maximum time to run 

## **Deep Networks for Algorithms**

### For Kinematics Input
![Screenshot from 2024-02-14 19-25-04](https://github.com/rafiqollective/thesis/assets/89272933/b8130389-55fd-441f-8465-d58e53807147)

### For Birdview Input
![Screenshot from 2024-02-14 19-26-59](https://github.com/rafiqollective/thesis/assets/89272933/15c4ae4f-f9fa-4b3b-a109-1294e4b1fa1e)

# RESULS

### **Performance Graphs**

![avegare_100](https://github.com/rafiqollective/thesis/assets/89272933/0c8a128f-5db6-4226-8065-db022594c656)
![episode_reward](https://github.com/rafiqollective/thesis/assets/89272933/181cbf88-4631-4834-a8ef-04b97084d89f)
![episode_len](https://github.com/rafiqollective/thesis/assets/89272933/913e36e3-ac57-49c0-894a-67c385a61563)


## Citation

If you find this work useful, please cite:

```bibtex
@INPROCEEDINGS{11329257,
  author={Osman Özturan, Mehmet Yaşar},
  booktitle={1. International Black Sea Scientific Research and Innovation Congress.}, 
  title={Comparative Analysis of SAC and DDQN in 2D Autonomous Vehicle Simulation: Insights From Grayscale Bird View And Kinematics Inputs}, 
  year={2023},
  keywords={Training;Soft Actor-Critic (SAC);Double Deep Q-Network;Navigation;Filtering;Deep reinforcement learning;Driver behavior;Autonomous vehicles;Tuning},
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

