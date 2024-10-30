def save_new_tokenizer():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
    tokenizer = processor.tokenizer
    print(len(tokenizer))
    # 新增token
    special_tokens_dict = {'additional_special_tokens': ['<|tokens|>', '<|to|>','<|be|>','<|added|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    processor.tokenizer = tokenizer
    print(len( processor.tokenizer))
    save_path = PROCESSOR_PATH
    processor.save_pretrained(save_path)

    return processor

def adjust_model_embedding(tokenizer):
    model = Qwen2AudioForConditionalGeneration.from_pretrained("/pretrained_model")
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained("/samll_model")
    return model


if __name__ == "__main__":

    tokenizer = save_new_tokenizer()
    model = adjust_model_embedding(tokenizer)