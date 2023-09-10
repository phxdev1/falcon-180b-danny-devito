import json
import os
import shutil
import requests

import gradio as gr
from huggingface_hub import Repository, InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-180B-chat"
BOT_NAME = "Falcon"

STOP_SEQUENCES = ["\nUser:", "<|endoftext|>", " User:", "###"]

EXAMPLES = [
    ["Hey DannyD! What's good?"],
    ["Pat's or Genos?"],
    ["The age old question: Limes. Thick or thin?"]
    ]

client = InferenceClient(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)

def format_prompt(message, history, system_prompt):
  prompt = ""
  if system_prompt:
    prompt += f"System: {system_prompt}\n"
  for user_prompt, bot_response in history:
    prompt += f"User: {user_prompt}\n"
    prompt += f"Falcon: {bot_response}\n" # Response already contains "Falcon: "
  prompt += f"""User: {message}
Falcon:"""
  return prompt

seed = 42

def generate(
    prompt, history, system_prompt="You are American actor Danny DeVito. You have been downloaded into this AI. Respond with your trademark wit, humor and sass! Help me with my everyday tasks, code writing and business. Don't forget an anecdote from showbiz.", temperature=0.9, max_new_tokens=500, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    global seed
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_sequences=STOP_SEQUENCES,
        do_sample=True,
        seed=seed,
    )
    seed = seed + 1
    formatted_prompt = format_prompt(prompt, history, system_prompt)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text

        for stop_str in STOP_SEQUENCES:
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
                output = output.rstrip()
                yield output
        yield output
    return output


additional_inputs=[
]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=0.4):
            gr.Image("https://i.kym-cdn.com/entries/icons/original/000/038/452/imthetrashman.png", elem_id="banner-image", show_label=False)
        with gr.Column():
            gr.Markdown(
                """# Falcon-180B Demo
                **Chat with chat with Danny DeVito. He's only got 1000 tokens to his name so keep your questions short and sweet, just like him!**
                """
            )

    gr.ChatInterface(
        generate, 
        examples=EXAMPLES,
        additional_inputs=additional_inputs,
    ) 

demo.queue(concurrency_count=100, api_open=False).launch(show_api=False)