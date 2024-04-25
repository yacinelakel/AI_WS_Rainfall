from transformers import RobertaForCausalLM, RobertaTokenizer

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text_gpt(prompt, model_name="gpt2", max_length=50):
    # Load pretrained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Generate text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95,
                            temperature=0.7)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text, output, tokenizer

def tokenize_text(text, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    return tokens