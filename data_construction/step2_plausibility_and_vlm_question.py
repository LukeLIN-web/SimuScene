from openai import OpenAI
import pandas as pd
import re
import os
import tqdm
# Initialize OpenAI client
client = OpenAI(
    api_key="xx"
)

def clean_response(text):
    # Remove <think> tags and their content
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def get_features(s):
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return s

# OpenAI models to use
models = [
    "gpt-4o"
]

def get_model_response(model, content, index):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0.7,
            max_tokens=16384
        )
        response = completion.choices[0].message.content
        cleaned_response = get_features(clean_response(response))
        return index, model, content, cleaned_response
    except Exception as e:
        print(f"Error getting response from {model} for row {index}: {str(e)}")
        return index, model, content, ""

def main():
    topic = ['gravity', 'elastic force','Linear Motion','Curvilinear Motion','Circular Motion','Relative Motion','Friction','Kinetic Energy','Potential Energy','Conservation of Mechanical Energy','Linear Momentum','Impulse','Conservation of Momentum','Simple Pendulum','Spring Oscillator','Damped Oscillation','Driven Oscillation','Resonance','Moment of Inertia','Angular Momentum','Torque','Rotational Motion','Pressure','Fluid Dynamics','Fluid Statics','Elastic Collision','Inelastic Collision','Temperature','Heat','Ideal Gas Law','Coulomb\'s Law','Electric Field','Electric Potential','Gauss\'s Law','Electric Current','Resistance','Ohm\'s Law','Capacitance','Inductance','Magnetic Field','Ampère\'s Law','Magnetic Flux','Faraday\'s Law of Electromagnetic Induction','Lenz\'s Law','Self-Inductance','Mutual Inductance','Reflection of Light','Refraction of Light','Lens','Interference of Light','Diffraction of Light','Polarization']
    for i in range(len(topic)):
        question_list = pd.read_csv(f'generated/questions_{i}.csv')
        questions = question_list["Questions"].tolist()
        levels = question_list["Level"].tolist()
        # Read interactions from CSV
        
        for j in tqdm.tqdm(range(0, len(question_list))):
            for model in models:
                questions_save = []
                
                content = f'''Determine whether the question "{questions[j]}" is physically plausible, internally consistent, macroscopically visualizable, and feasible to represent using Python code.

If it is not, return only:

no

If it is plausible, generate a set of True/False verification questions in JSON format to help verify whether an input video accurately reflects the user description. The user description is:

"{questions[j]}"

Base the questions only on what is described in the user content. Ignore implementation details and avoid adding extra assumptions.

The goal is to help a human reviewer quickly check whether the video is faithful to the user's request. The questions should be comprehensive enough to determine whether the motion process follows the laws of physics, such as whether the direction of motion after a collision is logical, whether a ball moves downward during free fall, or whether the motion exceeds the range of the container.

Instructions:

1. Generate at least five clear and focused True/False questions that are comprehensive enough to determine whether the motion process follows the laws of physics.
2. Each question must directly verify a visually observable element from the description.
3. Focus on key aspects:
   - Object motion or trajectories.
   - Number of bounces or interactions.
   - Triggered events or sensors.
   - Boundary conditions.
   - Visual clarity or layout.
4. Avoid abstract, subjective, or inferred content. Avoid time-related questions, such as speed, acceleration, and distance. Do not guess details that are not in the description.
5. Do not reference Python code. Focus only on what the video is supposed to show.
6. Do not focus on unrelated appearance questions, such as whether a plane is yellow or a ball is steel.
7. Do not focus on text displayed in the video, such as whether the ball's weight matches a label or whether the video shows a length value.
8. Avoid specific numeric values, such as 70 cm, 60 degrees, or 2.5 seconds.
9. Avoid questions that are not related to the motion process, such as whether the red ball stands out against the background, whether the inclined plane surface is polished, or whether the setup is brightly illuminated.

Output format:

Return a JSON list of questions. Each item must contain:
- "id": an integer starting from 1
- "question": the verification question
- "answer": the answer to the question
- "type": "true_false"

Example:

[
  {{
    "id": 1,
    "question": "The ball begins at the top of the inclined plane.",
    "answer": "True",
    "type": "true_false"
  }},
  {{
    "id": 2,
    "question": "The ball moves downward along the inclined plane.",
    "answer": "True",
    "type": "true_false"
  }},
  {{
    "id": 3,
    "question": "The ball remains within the visible container boundary throughout the motion.",
    "answer": "True",
    "type": "true_false"
  }}
]

Return only "no" or the JSON list.
'''
                index, model, content_return, response = get_model_response(model, content, i)
                response_str = str(response)

                if len(response_str) < 10 or "no" in response_str.lower():
                    continue
                else:
                    vision_questions = response.replace("Yes.", "")
                    questions_save.extend([
                        [questions[j], levels[j], vision_questions]
                    ])
                file_path = f'generated/filtered_questions_{i}.csv'
                df = pd.DataFrame(questions_save, columns=['Questions','Level','VisionQuestions'])
                if not os.path.exists(file_path):
                    df.to_csv(file_path, index=False, quoting=1)
                else:
                    df.to_csv(file_path, mode='a', header=False, index=False, quoting=1)
                    

if __name__ == "__main__":
    main() 