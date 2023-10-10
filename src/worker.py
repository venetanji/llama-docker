import torch
import transformers
import asyncio
from transformers import AutoTokenizer
from gptqllama.llama_inference import load_quant, get_llama
from config import Redis
import yaml
redis = Redis()

config = yaml.safe_load(open("src/config.yml"))

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"


model_path = 'TheBloke/Llama-2-13B-Chat-GPTQ'
model = load_quant(model_path,
           "/models/Llama-2-13B-Chat-GPTQ/gptq_model-4bit-128g.safetensors",
           4, 128)

model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

def as_instance(prompt):
    return B_INST + prompt + E_INST

system_prompt = B_SYS + config['system'] + E_SYS

def as_complete(prompt):
    return BOS + prompt + EOS
    

async def main():
    redis_connection = await redis.create_connection()
    while True:
        completion = await redis_connection.xread({'chat': '$'}, None, 0)
        # if it's an assistant message, skip it
        if completion[0][1][0][1][b'role'].decode('utf-8') == 'assistant':
            continue
        # parse json completion
        
        print(completion)
        conversation = await redis_connection.xrange('chat', '-', '+')
        for idx, message in enumerate(conversation):
            instance = message[1][b'message'].decode('utf-8') 
            role = message[1][b'role'].decode('utf-8')
            print(role, instance)
            if  role == 'assistant':
                prompt += instance
            elif role == 'user':
                if idx == 0:
                    prompt = as_instance(system_prompt + instance)
                prompt += as_instance(instance)
            
            
        print("Prompt", prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=True,
                min_length=40,
                max_length=1000,
                top_p=0.95,
                temperature=0.8,
            )
        response = tokenizer.decode([el.item() for el in generated_ids[0]])
        # return the second last instance between E_INST and EOS
        bot_response = response.split(E_INST)[-1].split(EOS)
        if len(bot_response) > 1:
            bot_response = bot_response[-2]
        else:
            bot_response = bot_response[0]
            print("Prompt incomplete...")
        bot_response.strip()
        print("Bot response", bot_response)
        await redis_connection.xadd('chat', {'role':'assistant', 'message': bot_response})
        print(response)

if __name__ == "__main__":
    asyncio.run(main())