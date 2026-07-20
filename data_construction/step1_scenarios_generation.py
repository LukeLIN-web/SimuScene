from openai import OpenAI
import pandas as pd
import re
import os, sys
import tqdm
# Initialize OpenAI client
client = OpenAI(
    api_key="xx"
)
os.makedirs("logs", exist_ok=True)

def clean_response(text):
    # Remove <think> tags and their content
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def get_features(s):
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return s

def extract_triple_single_quote_code(text):
    # Regex pattern to match content inside ''' '''
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def extract_triple_single_quote_json(text):
    # Regex pattern to match content inside ''' '''
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

# OpenAI models to use
models = [
    "gpt-4o"
]

def get_model_response(model, content, index, filter_content=True):
    try:
        system_prompt = f"""You are an expert computational physicist specializing in scientific visualization and simulation.
You are also an excellent programmer.
Your expertise includes creating educational physics simulations that effectively communicate complex physical phenomena to diverse audiences.
"""
        if model == "o4-mini":
            completion = client.chat.completions.create(
                model=model,
                reasoning_effort="high",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ]
                
            )
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=0.7,
                max_tokens=16384
            )
        response = completion.choices[0].message.content
        if filter_content:
            cleaned_response = get_features(clean_response(response))
        else:
            cleaned_response = response
        return index, model, content, cleaned_response
    except Exception as e:
        print(f"Error getting response from {model} for row {index}: {str(e)}")
        return index, model, content, ""


def get_conditions(topic):
    conditions = []
    content = f"""Generate a diverse list of unique and realistic conditions that can be added to or used to extend the topic "{topic}" in everyday physical scenarios. Each condition must be:

- Visually clear and easy to program.
- Distinct and non-redundant.
- Physically meaningful at the human-observable scale.
- Free of temperature-related, biological, weather-related, micro-scale, or instantaneous effects that are difficult to observe or simulate.

Return the result as a plain list separated by the "@" symbol.

Do not include atomic-level effects, microscopic interactions, or abstract forces that cannot be clearly visualized.

Each condition should increase the complexity or variety of the "{topic}"-related scenario in a macroscopically observable and visualizable way.

Example:

confining the object's motion within a container@adding multiple moving objects@introducing rotational dynamics@applying time-dependent changes
"""

    for model in models:
        index, model, content_return, response = get_model_response(model, content, 0)
        conditions.extend(response.split('@'))
    os.makedirs('generated/conditions/', exist_ok=True)
    df = pd.DataFrame(conditions, columns=['Conditions'])
    df.to_csv(f'generated/conditions/{topic.replace(" ", "_")}.csv', index=False)
    return conditions

def get_basicquestions(topic):
    basicquestions = []
    content = f"""Generate as many distinct and simple visualizable physics problems as possible related to the topic of "{topic}". Each problem should describe a clearly different type of motion or phenomenon, using basic geometric shapes, such as a ball, cube, beam, or dot, to represent objects.

Avoid repeated or similar motion types. For example, "a ball falls down" and "an apple dropped from a tree" represent the same motion and should not both be included.

Exclude scenarios involving temperature, biology, weather, microscopic physics, or effects that are not easily observable over time.

Return only a plain list of concise problem descriptions, separated by the "@" symbol.

For example:

a ball falling freely under gravity@a beam of light striking a horizontal surface@red liquid diffusing into water

Do not include duplicates or closely related variations.
"""
    for model in models:
        index, model, content_return, response = get_model_response(model, content, 0)
        basicquestions.extend(response.split('@'))
    os.makedirs('generated/basicquestions/', exist_ok=True)
    df = pd.DataFrame(basicquestions, columns=['BasicQuestions'])
    df.to_csv(f'generated/basicquestions/{topic.replace(" ", "_")}.csv', index=False)
    return basicquestions

def main():
    # physics topics
    topic = ['gravity', 'elastic force','Linear Motion','Curvilinear Motion','Circular Motion','Relative Motion','Friction','Kinetic Energy','Potential Energy','Conservation of Mechanical Energy','Linear Momentum','Impulse','Conservation of Momentum','Simple Pendulum','Spring Oscillator','Damped Oscillation','Driven Oscillation','Resonance','Moment of Inertia','Angular Momentum','Torque','Rotational Motion','Pressure','Fluid Dynamics','Fluid Statics','Elastic Collision','Inelastic Collision','Temperature','Heat','Ideal Gas Law','Coulomb\'s Law','Electric Field','Electric Potential','Gauss\'s Law','Electric Current','Resistance','Ohm\'s Law','Capacitance','Inductance','Magnetic Field','Ampère\'s Law','Magnetic Flux','Faraday\'s Law of Electromagnetic Induction','Lenz\'s Law','Self-Inductance','Mutual Inductance','Reflection of Light','Refraction of Light','Lens','Interference of Light','Diffraction of Light','Polarization']
    # topic = ['Reflection of Light', 'elastic force', 'Kinetic Energy']
    short_topic = []
    total=0
    for i in range(len(topic)):
        file_path = f'generated/questions/questions_{topic[i].replace(" ", "_")}.csv'
        if not os.path.exists(file_path):
            short_topic.append(topic[i])
        else:
            df = pd.read_csv(file_path)
            num_rows = len(df)
            total += num_rows
            print(f'topic:{topic[i]}, questions: {num_rows}')
    print('total:', total)
    print(short_topic)
    topic = short_topic
    total_questions = []
    question_num = 0
    for i in tqdm.tqdm(range(len(topic))):
        sys.stdout = open(f"logs/output_{topic[i].replace(' ', '_')}.txt", "w")
        condition_exist = os.path.exists(f'generated/conditions/{topic[i].replace(" ", "_")}.csv')
        question_exist = os.path.exists(f'generated/basicquestions/{topic[i].replace(" ", "_")}.csv')
        total_questions.append(0)
        # get conditions
        if condition_exist:
            conditions = pd.read_csv(f'generated/conditions/{topic[i].replace(" ", "_")}.csv')
        else:
            conditions = get_conditions(topic[i])
        # get basic questions
        if question_exist:
            basicquestions_file = pd.read_csv(f'generated/basicquestions/{topic[i].replace(" ", "_")}.csv')
            basicquestions = basicquestions_file.iloc[:, 0].tolist()
        else:
            basicquestions = get_basicquestions(topic[i])
        if not condition_exist or not question_exist:
            continue
        questions_save = []
        # generate questions
        for model in models:
            for question_index in tqdm.tqdm(range(len(basicquestions))):
                all_conditions = ','.join(conditions)
                for cond_num in range(3):
                    total_questions[i] += len(questions_save)
                    questions_save = []
                    question_num = 10
                    if cond_num > 2:
                        question_num = 5
                    content = f'''Generate {question_num} unique, logically consistent, and visually descriptive physics problems based on "{basicquestions[question_index]}". Use real-world, Earth-based scenarios that occur within a closed container or controlled setup. Avoid space-based or abstract environments.

Each problem must satisfy the following requirements:

1. Incorporate {cond_num + 1} distinct and non-overlapping conditions selected from {all_conditions}, with specific numeric parameter values.
2. Be described as a self-contained paragraph featuring a unique physical process that lasts more than 10 seconds.
3. Be fully visualizable using basic colored shapes and lines to represent key elements, such as a ball, inclined plane, liquid, or pulley.
4. Provide precise physical settings, including object mass, slope angle, container dimensions, surface materials, and environmental conditions.
5. Avoid motion contradictions, such as an object slowing down after release under gravity without resistance.
6. Avoid repeating motion patterns or setup logic across the generated problems.
7. Do not reference video rendering, animation, or simulation.
8. Only return the generated problems, and do not include any other information.

Return the results in the following format:

-1- First problem.
-2- Second problem.
-3- Third problem.
-4- ...
'''

                    _, model, _, response = get_model_response(model, content, i)
                    response_list = re.split(r"-(\d+)-", response)

                    sub_question_list = [x for x in response_list if len(x) > 5]

                    for sub_index in range(len(sub_question_list)):
                        os.makedirs(f'video/{topic[i].replace(" ", "_")}', exist_ok=True)
                        video_path = f'video/{topic[i].replace(" ", "_")}/{question_num}'
                        questions_save = []
                        questions_save.append(
                            [sub_question_list[sub_index], cond_num+1]
                        )
                        file_path = f'generated/questions/questions_{topic[i].replace(" ", "_")}.csv'
                        df = pd.DataFrame(questions_save, columns=['Questions', 'Level'])
                        if not os.path.exists(file_path):
                            df.to_csv(file_path, index=False, quoting=1)
                        else:
                            df.to_csv(file_path, mode='a', header=False, index=False, quoting=1)
                        question_num += 1
                print(f'topic: {i}, basic {question_index} finished')                     
        print(f'topic: {topic[i]} finished')
    for i in range(len(total_questions)):
        print(f'topic: {topic[i]}, questions: {total_questions[i]}')
    print(f'total questions: {sum(total_questions)}')

if __name__ == "__main__":
    main() 