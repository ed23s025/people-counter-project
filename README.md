# People Counter Using YOLOv8 & ByteTrack

This project detects people in a video, tracks them using unique IDs, and counts **Entry** (when moving from top to bottom across a line) and **Exit** (when moving from bottom to top).  
It uses YOLOv8 for detection and ByteTrack for tracking.  
The counting logic is based on a single green horizontal line in the center of the video. 

---

## Features

Person detection using YOLOv8  
Multi-person tracking with ByteTrack  
Counts Entry (top → bottom) and Exit (bottom → top)  
Only one green center line used (no red line)  
Output saved automatically as `output.mp4`  

---

## Project Structure

├── counter.py        
├── input.mp4          
├── output.mp4         
├── README.md          
└── requirements.txt   



---

## ⚙️ Installation & Setup
Install Python Libraries

```bash
pip install ultralytics opencv-python numpy


python -m venv env
env\Scripts\activate         
pip install ultralytics opencv-python numpy


##Run the code
python counter.py

