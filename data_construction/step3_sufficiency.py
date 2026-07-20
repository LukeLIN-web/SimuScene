"""
Simple LLM Client Usage Examples

This shows how to use the LLMClient from utils.py in your own projects.
The LLMClient already handles server discovery, load balancing, rate limiting, and error handling.
"""

import asyncio
from utils import LLMClient
import os
import json
import re
from tqdm import tqdm
import sys

Model_name = "deepseek-ai/DeepSeek-R1-0528"
savejsonl = True
json_arr = []

def clean_response(text):
    # Remove <think> tags and their content
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
def extract_triple_single_quote_json(text):
    # Regex pattern to match content inside ''' '''
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def create_prompt(question, all_vlm_questions):
    return f"""You are an annotator.

You are given a motion description:

{question}

You are also given a set of related visual verification questions:

{all_vlm_questions}

Assume that the answers to all of these questions are "Yes".

Determine whether the questions in {all_vlm_questions} are sufficient to fully judge the entire motion described in {question}. Your response should be either "Yes" or "No".

If your answer is "Yes", summarize the key noun phrases in the motion description and describe their relationships. For example, "a red ball is moving toward a blue ball".

The output should be in JSON format with the following structure:

{{
  "enableAnnotator": "Yes or No",
  "summarize": "A red ball is moving toward a blue ball.",
  "vlm_questions": [
    {{
    "index": "index number",
    "question": "question1",
    "weight": "0-1 scale, where 0 is not important and 1 is very important"
    }}
  ]
}}

"""

async def get_response(client, question, id, level, all_questions):
    if type(all_questions) == str and all_questions.strip() == 'no':
        return
    folder_id = int(id) // 2500
    try:
        if type(all_questions) == str:
            vlm_questions = json.loads(all_questions)
        else:
            vlm_questions = all_questions
    except:
        fixed = all_questions.replace('\\(', '\\\\(').replace('\\)', '\\\\)')
        vlm_questions = json.loads(fixed)
    
    all_questions = ''
    for j in range(len(vlm_questions)):
        all_questions += f'{j+1}. {vlm_questions[j]["question"]}'
    messages = [
        {
            "role": "user",
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "content": 
                [{
                    "type": "text", 
                    "text": create_prompt(question, all_questions)
                }]
        }]
    if savejsonl:
        json_arr.append({
            "messages": messages,
            "id": id,
            "question": question,
            "level": level,
        })
        return
    responses = await client.process_all_inputs([messages], model=Model_name)
    try:
        result = clean_response(responses[0][0])
        try:
            json.loads(result)
        except:
            result = extract_triple_single_quote_json(result)
        json_folder = f'verify/batch_{folder_id}/video_{id}'
        os.makedirs(json_folder, exist_ok=True)
        save_json = {
            "question": question,
            "Level": level,
            "vlm_questions": result,
            "id": id,
        }
        with open(f'{json_folder}/video_{id}.json', 'w', encoding='utf-8') as f:
            json.dump(save_json, f, indent = 4, ensure_ascii=False)
    except:
        print('error:', responses)

async def main():
    client = LLMClient(model=Model_name, timeout_seconds=10000, temperature=0.5)
    with open("conversations_response_new_with_level_q.jsonl", 'r') as f:
        for i, row in tqdm(enumerate(f)):
            item = json.loads(row)
            msg = item['messages']
            try:
                await get_response(client, msg[1]['content'].split('\n')[4], item['id'], item['Level'], item['vision_questions'])
            except:
                print('error:', i)
    if savejsonl:
        with open("run.jsonl", "w", encoding="utf-8") as f:
            for item in json_arr:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
     
def process_jsonl():
    with open("run_responses.jsonl", "r", encoding="utf-8") as f:
        for i, row in tqdm(enumerate(f)):
            item = json.loads(row)
            folder_id = int(item['id']) // 2500
            id = item['id']
            level = item['level']
            
            for response in item['responses']:
                result = clean_response(response)
                try:
                    json.loads(result)
                except:
                    result = extract_triple_single_quote_json(result)
                if type(result) == list:
                    if len(result) == 0:
                        result = '{"enableAnnotator": "No"}'
                    else:
                        result = result[0]
                result = json.loads(result)
                json_folder = f'result/verify/batch_{folder_id}/video_{id}'
                os.makedirs(json_folder, exist_ok=True)
                save_json = {
                    "question": item['question'],
                    "Level": level,
                    "vlm_questions": result,
                }
                with open(f'{json_folder}/video_{id}.json', 'w', encoding='utf-8') as f:
                    json.dump(save_json, f, indent = 4, ensure_ascii=False)




if __name__ == "__main__":
    if sys.argv[1] == 'main':
        print('get run.jsonl')
        asyncio.run(main())
    elif sys.argv[1] == 'process_jsonl':
        print('process_jsonl')
        process_jsonl()
