from transformers import AutoTokenizer
import transformers
import torch
import sys

model = "meta-llama/Llama-2-7b-chat-hf"
access_token = sys.argv[1]

print(access_token)

# Define model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    token=access_token
)

# Run inference using pipeline
sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

# Print to file
with open("generated_70b_output.txt", "w") as f:
    for seq in sequences:
        f.write(seq["generated_text"])
        f.write("\n\n")
