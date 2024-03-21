import clip
from transformers import AutoTokenizer

########################################################################################
#
# Tokenizer
#
########################################################################################

def clip_tokenize(text):
    """Tokenize text with CLIP tokenizer"""
    tok = clip.tokenize(text, truncate=True) 
    return tok.squeeze()

def bert_tokenize(text, tokenizer, **kwargs):
    """
    Tokenize text with Hugging Face tokenizer

    :param str text: Text to tokenize
    :param tokenizer: Hugging Face transformer tokenizer
    :param kwargs: Arguments for tokenizer 

    :example:
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> bert_tokenize("Hello world", tokenizer)
    """
    return tokenizer(text, **kwargs)
