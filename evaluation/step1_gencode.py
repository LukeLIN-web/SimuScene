import json
import pandas as pd
import os


def get_prompt(content):
    system_prompt = f"""You are an expert computational physicist specializing in scientific visualization and simulation.
You are also an excellent programmer.
Your expertise includes creating educational physics simulations that effectively communicate complex physical phenomena to diverse audiences.
"""

    
    user_prompt = f"""Your task is to write a Python script that generates an educational video simulating a physical process. The video will be used in academic settings to help students better understand and visualize physics concepts.

Given a textual description of a physical process:

"{content}"

Write a Python script that simulates this process and outputs a video saved as:

"name.mp4"

Here, "name" is a user-defined parameter passed when running the script.

Requirements for the video output:

1. Use clear and distinct colors to represent different objects, trajectories, or forces.
2. Overlay a real-time timestamp that updates continuously throughout the simulation.
3. Display all relevant parameter values, such as gravity, speed, and angle, clearly on the screen.
4. Ensure the camera view is wide enough to fully capture the entire motion, adjusting dynamically if needed. Use the best view for the simulation so that the viewer can see the whole process.
5. Provide smooth and continuous animation at a consistent frame rate of 30 FPS.
6. Maintain a clean, uncluttered visual style with minimal distractions and a neutral background.
7. Keep the video duration between 10 and 20 seconds, slow enough to allow viewers to observe and understand the key transitions.
8. Save the output as an MP4 video in a suitable resolution of at least 360p.
9. When the process finishes, the video should also finish.
10. Use OpenCV to generate the video, and ensure that the code is correct, complete, and runnable without errors.

Focus on clarity, interpretability, and visual appeal to make the video intuitive and easy to understand for both technical and non-technical audiences.

Physics simulation:

- Implement precise physical equations.
- Use appropriate time steps for smooth motion.
- Include relevant force vectors and trajectories.

The final code should:

1. Initialize all necessary libraries and variables.
2. Set up the video writer with specified parameters.
3. Implement the physics calculations.
4. Create and save the animation.
5. Include error handling and resource cleanup.

Ensure that the code follows PEP 8 style guidelines and includes comments explaining key components. The simulation should prioritize educational value while maintaining scientific accuracy.

Instructions:

1. Output only the complete Python code. Do not include explanations outside the code. For example, the code may start with:

import cv2
import numpy as np

2. The code should install any dependencies within the script if needed.
3. The code should run without errors.
4. The code should be complete and self-contained.
5. The code should be correct and physically accurate.
6. The code should be efficient and optimized.
7. The code should be easy to understand and modify.
"""
    
    return system_prompt, user_prompt

output_file = open("simuscene_val_getcode.jsonl", "w", encoding="utf-8")
questions = []
for json_file in os.listdir('human_modify/val/'):
    data = json.load(open(os.path.join('human_modify/val/', json_file), 'r', encoding='utf-8'))
    video_index = json_file.split('_')[1].replace('.json', '')

    system_prompt, user_prompt = get_prompt(data['question'])
    conversation = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "video_index": video_index
    }
    output_file.write(json.dumps(conversation, ensure_ascii=False) + "\n")


output_file.close()


