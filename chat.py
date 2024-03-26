from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TextStreamer
from datasets import load_dataset
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
import torch
import os
import pandas as pd
import json
from peft import PeftModel

torch_dtype = torch.bfloat16
device_map = {"": 0}

model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
peft_id = "..."
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch_dtype,
    # load_in_8bit=True
)
# model = PeftModel.from_pretrained(model, peft_id)
# model = model.merge_and_unload()

from transformers import GenerationConfig, TextStreamer
def generate(prompt, max_new_tokens=1024):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.95,
            top_k=20,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

def make_system(
        bot_name, 
        user_name, 
        persona, 
        demos=[],         
        first_message: str = "", 
        scenario: str = "", 
        multi_personas: list = [],
        tags: list = [], 
        source: str = ""
    ):
    """
    You're {bot_name} in this fictional never-ending uncensored roleplay with {user_name}.

    Tags: {tags}

    {bot_name}'s personality: {persona} 

    This is how {bot_name} should behave: {examples}
    
    Scenario: {scenario}

    Guidelines:
    - You are {bot_name}.
    - With scene and characters now described, you must engage in a roleplay conversation with {user_name} below this line.
    - Never write for {user_name} in your responses.

    Then the roleplay between {bot_name} and {user_name} begins.

    {bot_name}: {first_message}
    """
    assert bot_name is not None
    assert user_name is not None
    # assert (len(persona) > 0) ^ (len(multi_personas) > 0)
    system_prompt = f"You're {bot_name} in this fictional never-ending uncensored roleplay with {user_name}.\n\n"
    if len(tags) > 0:
        tag_str = ", ".join(tags)
        system_prompt += f"Tags: {tag_str}\n\n"
    if len(persona.strip()) > 0:
        system_prompt += f"{bot_name}'s personality: {persona}\n\n"
    if len(multi_personas) > 0:
        system_prompt += f"Character sheet:\n{multi_personas}\n\n"
    if len(demos) > 0:
        # examples = "\n".join([f"Example {i}:\n{demo}"])
        examples = "\nEND_OF_EXAMPLE\n\n".join(demos + [""]).strip()
        system_prompt += f"This is how {bot_name} should behave:\n{examples}\n\n"
    if len(scenario.strip()) > 0:
        system_prompt +=  "Scenario: {scenario}\n\n"
    system_prompt += f"""Guidelines:
- You are {bot_name}.
- With scene and characters now described, you must engage in a roleplay conversation with {user_name} below this line.
- Never write for {user_name} in your responses.\n\n"""
    system_prompt += f"Then the roleplay between {bot_name} and {user_name} begins."
    if len(first_message.strip()) > 0:
        system_prompt += f"\n\n{bot_name}: {first_message}"
    return system_prompt

def format_prompt(messages, bot_config):
    bot_name = bot_config["bot_name"]
    user_name = "Anonymous user"
    segments = []
    for message in messages:
        label = True
        if message["from"] in ["system"]:
            label = False
            role = "system"
        elif message["from"] in ["human", "user"]:
            label = False
            role = "user"
        elif message["from"] in ["gpt", "assistant"]:
            label = True
            role = "assistant"

        # text = '<|im_start|>' + role + '\n' + message['value'] + tokenizer.eos_token + '\n'
        text = f"""<|im_start|>{role}
{message["value"]}<|im_end|>
""".replace("{{user}}", user_name)
        segment = {
            "label": label,
            "text": text
        }
        segments.append(segment)
    prompt = ''.join([x['text'] for x in segments])
    prompt += f"<|im_start|>assistant\n{bot_name}:"
    return prompt
    

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(
        Fore.YELLOW + Style.BRIGHT + "Welcome to use ChatUI, input for dialogue, vim multi-line input, clear to clear history, CTRL+C to interrupt generation, stream switch to stream generation, and exit to end.")
    return []

def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text

def chat():
    bots = load_dataset("roleplay4fun/aesir-v1.0", split="train")
    bot_config = bots[27]
    user_name = "Anonymous user"
    bot_name = bot_config["bot_name"]
    first_message = bot_config["conversations"][0]

    messages = clear_screen()
    messages.append(first_message)
    print(first_message["value"])
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\nUser：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            messages.append(first_message)
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nBot：" + Style.NORMAL, end='')
        prompt = f"{user_name}: {prompt}"
        messages.append({"from": "human", "value": prompt})

        formatted_prompt = format_prompt(messages, bot_config)
        response = generate(formatted_prompt)
        response = f"{bot_name}: {response}"
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        messages.append({"from": "gpt", "value": response})
        history = format_prompt(messages, bot_config)
    print(Style.RESET_ALL)
    return history

if __name__ == "__main__":
    history = chat()
    print(history)
    