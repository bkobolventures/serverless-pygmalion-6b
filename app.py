from transformers import GPTJForCausalLM, AutoTokenizer, pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer
    global generator

    print("loading to CPU...")
    model = GPTJForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    # tokenizer = GPT2Tokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
    tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    

    defaults = {
        "max_new_tokens": 75,
        "top_k": 3,
        "top_p": 0.75,
        "temperature": 0.9
    }

    model_params = defaults | model_inputs
    # Decode output tokens
    output_text = generator(prompt, **model_params)

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
