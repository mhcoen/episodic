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

save_file_name = '..json'

PATH = '/.json'
wiki= wikipediaapi.Wikipedia('', 'en', extract_format=wikipediaapi.ExtractFormat.WIKI)

with open(PATH) as f:
    data = json.load(f)
    
keys = list(data.keys())
used_set = set()
path = []
relation = []

for i in tqdm(range(data_num)):
    tmp_path = []
    tmp_relation = []
    pick_root_count = 0
    while pick_root_count < 5:
        current_sub = random.choice(keys)
        if current_sub not in used_set:
            break
        pick_root_count += 1     
    
    path_length = 0
    while path_length < 4:
        try_count = 0
        while try_count < 5:
            try:
                random_object = random.choice(data[current_sub])
                current_obj = random_object[0][2]
                current_relation = random_object[1]
            except IndexError:
                random_object = random.choice(data[current_sub][0])
                current_obj = random_object[0][2]
                current_relation = random_object[1]
                
            if current_obj not in used_set:
                break
            try_count += 1
        if try_count == 5 or current_sub == current_obj:
            break
            
        if path_length > 0:
            tmp_path.append(current_obj)
            used_set.add(current_obj)
            tmp_relation.append(current_relation)
            
        else:
            tmp_path.append(current_sub)
            used_set.add(current_sub)
            tmp_path.append(current_obj)
            used_set.add(current_obj)
            tmp_relation.append(current_relation)
        
        
        if current_obj in keys:
            current_sub = current_obj
        else:
            break
        path_length += 1
    
    if tmp_path != []:
        path.append(tmp_path)
        relation.append(tmp_relation)
        
path_len = []
relation_len = []

for i in range(len(path)):
    path_len.append(len(path[i]))

print("Path len : ", statistics.mean(path_len))

for i in range(len(path)):
    if len(path[i]) != len(relation[i])+1:
        print("error")
        print(i)

answers_list = []
passages = []
topic = []
sent_seg = []
#for i in range(len(path)):

print("---------------PASSAGE GENERATION----------------")
for i in tqdm(range(len(path))):
    answers = []
    current_passage_list = []
    passage = ''
    tmp_sent_seg = []
    current_topic = []
    flag = True

    for j in range(len(path[i])):
        while True:
            try:
                current_passage_list.append(nltk.sent_tokenize(wiki.page(path[i][j]).text))
                break
            except Exception as e:
                print(e)
                print("Sleeping 10 secs....")
                time.sleep(10)
        text = current_passage_list[j]
        lang_code, _ = langid.classify(str(text))

        if len(text) == 0:
            flag = False
            break
        if lang_code != 'en':
            flag = False
            break
    if not flag:
        continue
        #print(len(text))
    for j in range(len(path[i])):
        text = current_passage_list[j]
        current_topic.append(path[i][j])
        use_sent = random.randint(3,6)
        for check_end in range(len(text)):
            if "Reference" in str(text[check_end]) or "reference" in str(text[check_end]):
                del text[check_end:]
                break

        if len(text) > use_sent:
            answers = answers + text[:use_sent]
            tmp_sent_seg.append(use_sent)
        else:
            answers = answers + text
            passage = passage + " ".join(text)
            tmp_sent_seg.append(len(text))

        if j == len(path[i])-1:
            continue
        else:
            answers = answers + [relation[i][j]]
            passage = passage +" "+ str(relation[i][j]) +" "

    if answers != []:
        #print(tmp_sent_seg)
        answers_list.append(answers)
        passages.append(passage)
        sent_seg.append(tmp_sent_seg)
        topic.append(current_topic)
            
save_data = {}
save_data['answers_list'] = answers_list
save_data['passages'] = passages
save_data['sent_seg'] = sent_seg
save_data['topic'] = topic
              
with open(save_file_name, 'w') as f:
    json.dump(save_data, f)