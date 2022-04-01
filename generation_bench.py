import torch

from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")

article = """
Justin Timberlake and Jessica Biel, welcome to parenthood. 
The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People. 
"Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports. 
The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both.
"""


# inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)

def do_generation(article, model, tokenizer, max_length, min_length):
    inputs = tokenizer.encode(article, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length)
    # length_penalty=2.0,
    # num_beams=4,
    # early_stopping=True)
    # just for debugging
    # print(tokenizer.decode(outputs))
    # print(outputs)
    return outputs


outputs = do_generation(article, model, tokenizer, 100, 40)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
rouge = Rouge()

print(rouge.get_scores(tokenizer.decode(outputs[0]), article))
