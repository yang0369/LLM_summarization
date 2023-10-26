from config import config


def get_num_of_tokens(text):
    return len(config.TOKENIZER.tokenize(text))
