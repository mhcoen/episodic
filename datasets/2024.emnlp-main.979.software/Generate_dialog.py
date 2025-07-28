import wikipediaapi
import json
import random
import nltk
nltk.download('punkt')

import statistics
import re
from tqdm import tqdm
import langid
import json
import copy
import csv
import openai
import time

save_num = 10

start = 0
end = 11000

load_file_name = './.json'
save_file_name = './.json'

apikeys = ['']
api_num = 0
openai.api_key = apikeys[api_num]
wiki= wikipediaapi.Wikipedia('', 'en', extract_format=wikipediaapi.ExtractFormat.WIKI)

with open(load_file_name) as f:
    data = json.load(f)
    
answers_list = data['answers_list']
passages = data['passages']
sent_seg = data['sent_seg']
topic = data['topic']

print("---------------Dialog GENERATION----------------")
              
generated_data = []
instruction = f"You are an automatic assistant that generates appropriate question based on the predefined answer.\n \
Generate a single question that is most suitable for the given dialog history and target answer. \n"

for i in tqdm(range(start,end)):
    topic_shift = []
    turn = 0
    for k in range(len(sent_seg[i])-1):
        turn = turn + sent_seg[i][k]+1
        topic_shift.append(turn)
    current_dialog = []
    tmp = ''
    for j in range(len(answers_list[i])):
        pattern = r'\([^)]*\)'
        
        answer = re.sub(pattern= pattern, repl='', string = answers_list[i][j])
        if j in topic_shift:
            current_topic = topic[i][topic_shift.index(j)]
            next_topic = topic[i][topic_shift.index(j) + 1]
            base = f"Please fill in only [BLANK] in the next dialogue.\n Note that the conversation topic has changed into {next_topic} from {current_topic}. \n\
    [START]\n "
            QA_pair = f"\
    A: [BLANK]\n \
    B: {answer}\n \
    [END]"
            PROMPT = instruction + base + tmp + QA_pair
            #print(PROMPT)
        else:
            base = f"Please fill in only [BLANK] in the next dialogue.\n \
    [START]\n "
            QA_pair = f"\
    A: [BLANK]\n \
    B: {answer}\n \
    [END]"
            PROMPT = instruction + base + tmp + QA_pair
            #print(PROMPT)
        messages = [{"role": "system", "content": PROMPT}]
        
        while True:
            cnt = 0
            try:
                response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                #temperature=0,
                max_tokens=150)
                break
                
            except Exception as e:
                cnt = cnt + 1
                print(e)
                print("Sleeping 10 secs....")
                time.sleep(10)
                if cnt > 50:
                    api_num = api_num+1
                    openai.api_key = apikeys[api_num]
                if api_num == len(apikeys):
                    print("PAYMENT ERROR")
                    break
                    
        if api_num == len(apikeys):
            print("PAYMENT ERROR")
            break
            
        question = response.choices[0].message.content
        question = question.replace("A: ","").replace('\\', '').replace('"','').strip() # remove A and else...
        
        tmp = tmp +  f"\nA: {question}\n B: {answer}"
        current_dialog.append({"question": question, "answer": answer})

    generated_data.append({'topic_shift': topic_shift,'topics': topic[i],'passage':passages[i],'dialog':current_dialog})
    
    if i%save_num==0:
        with open(save_file_name, 'w') as f:
            json.dump(generated_data, f)
            
with open(save_file_name, 'w') as f:
    json.dump(generated_data, f)              