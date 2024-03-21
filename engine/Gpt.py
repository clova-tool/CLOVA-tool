import openai
from openai import OpenAI
import requests
import json
import time
import ruamel.yaml as yaml


LLM_config_path='configs/LLM_config.yaml'
LLM_config= yaml.load(open(LLM_config_path, 'r'), Loader=yaml.Loader)


def get_eval(content, llama_generator):


    count=0
    while(1):
        if count>10:
            content_output='some errors'
            break
        try:
            client = OpenAI(
                base_url=LLM_config['OpenAI']['base_url'],
                api_key=LLM_config['OpenAI']['api_key'],
            )
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
            )
            content_output=response.choices[0].message.content
            break
        except:
            count=count+1
            continue


    print ('!!!!!!!!!!!!!!get_eval!!!!!!!!!!!!!!')
    print ('content!!!',content)

    print ('content_output!!!',content_output)
    print ('!!!!!!!!!!!!!!get_eval!!!!!!!!!!!!!!')


    content_output=content_location(content_output)
    content_output = content_output.lstrip('\n').rstrip('\n')  

    return content_output


def content_location(input):

    index_s=input.find('\n')

    if index_s>=2:
        return input[:index_s+1]    
    else:
        return input


