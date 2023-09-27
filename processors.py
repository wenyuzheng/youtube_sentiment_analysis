from transformers import pipeline
import language_tool_python

# stdlib
import re

# Spelling checker for autocorrection
SPELL_CHECKER = {
    "en": language_tool_python.LanguageTool('en-US'),
    "es": language_tool_python.LanguageTool('es-ES')
}

#########################################
# Loaders
#########################################

def load_model_pipeline(task, model_path):
    return pipeline(task, model=model_path, tokenizer=model_path)

def load_language_classifier():
    return load_model_pipeline('text-classification', "papluca/xlm-roberta-base-language-detection")

def load_xlmr_model():
    return load_model_pipeline("sentiment-analysis", "./sentiments/model-xlmr")

def load_mbert_model():
    return load_model_pipeline("sentiment-analysis", "./sentiments/model-mbert")


#########################################
# Sanitizers
#########################################

EMOJI_PATTERNS = re.compile(
  "["
  u"\U0001F600-\U0001F64F"  # emoticons
  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
  u"\U0001F680-\U0001F6FF"  # transport & map symbols
  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
  u"\U00002500-\U00002BEF"  # chinese char
  u"\U00002702-\U000027B0"
  u"\U00002702-\U000027B0"
  u"\U000024C2-\U0001F251"
  u"\U0001f926-\U0001f937"
  u"\U00010000-\U0010ffff"
  u"\u2640-\u2642"
  u"\u2600-\u2B55"
  u"\u200d"
  u"\u23cf"
  u"\u23e9"
  u"\u231a"
  u"\ufe0f"  # dingbats
  u"\u3030"
  "]+", flags=re.UNICODE)

def remove_emoji(text, verbose=False):
    """
        Args:
            text: text before cleaning
        Returns:
            text: text without emojis
    """
    if verbose: print("[remove_emoji] original:", text)
    text = re.sub(EMOJI_PATTERNS, '', text)
    if verbose: print("[remove_emoji] processed:", text)
    return text

def remove_irrelevant_char(text, verbose=False):
    """
        Args:
            text: text before cleaning
        Returns:
            text: text without irrelevant characters
    """
    if verbose: print("[remove_irrelevant_char] original:", text)
    text = re.sub(r"[^a-záéíóúñüA-ZÁÉÍÓÚÑÜ\!'\"\.\?\,]", " ", text).strip()
    if verbose: print("[remove_irrelevant_char] processed:", text)
    return text

def remove_urls(text, verbose=False):
    """
        Args:
            text: text before cleaning
        Returns:
            text: text without URLs
    """
    if verbose: print("[remove_urls] original:", text)
    return re.sub(r'https?://\S+|www\.\S+', "", text)
    if verbose: print("[remove_urls] processed:", text)
    return text

def normalize_spaces(text, verbose=False):
    """
        Args:
            text: text before cleaning
        Returns:
            text: text with normalised spaces
    """
    if verbose: print("[normalize_spaces] original:", text)
    text = re.sub(r"(\s{2,}|\\r)", ' ', text).strip()
    if verbose: print("[normalize_spaces] processed:", text)
    return text

def remove_new_lines(text, verbose=False):
    """
        Args:
            text: text before cleaning
        Returns:
            text: text without new line characters
    """
    if verbose: print("[remove_new_lines] original:", text)
    text = re.sub(r"\n", " ", text)
    if verbose: print("[remove_new_lines] processed:", text)
    return text

def normalize_repeating_characters(text, verbose=False):
    """
        Args:
            text: text before cleaning
        Returns:
            text: text with normalised repeating characters
    """
    if verbose: print("[normalize_repeating_characters] original:", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if verbose: print("[normalize_repeating_characters] processed:", text)
    return text

#########################################
# Transformers
#########################################

def autocorrect_comment(text, lang, verbose=False):
    autocorrected_text = SPELL_CHECKER[lang].correct(text)
    if verbose: print("[autocorrect_comment] origianl:", text)
    if verbose: print("[autocorrect_comment] autocorrect:", autocorrected_text)
    return autocorrected_text

def preprocess_pipeline(text, verbose=False):
    if verbose: print("[preprocess_pipeline] original:", text)
    processed = remove_new_lines(text, verbose=verbose)
    processed = remove_emoji(processed, verbose=verbose)
    processed = remove_urls(processed, verbose=verbose)
    processed = normalize_spaces(processed, verbose=verbose)
    processed = remove_irrelevant_char(processed, verbose=verbose)
    processed = normalize_repeating_characters(processed, verbose=verbose)
    if verbose: print("[preprocess_pipeline] processed:", processed)
    return processed

#########################################
# Qualifiers
#########################################

def text_length(text):
    return len(text) > 3

def language_qualifier(text, classifier):
    """
        Args:
            text: text requiring language recognition
            classifier: a text language identifier
        Returns:
            A boolean whether the text is in either English or Spanish
            lang: language of the text
    """
    lang = classifier(text)[0]["label"]
    return (lang in ['en', 'es'], lang)

def qualify(text, classifier):
    """
        Args:
            text: text to be qualified
            classifier: a text language identifier
        Returns:
            A boolean whether the text is qualified or not
            lang: language of the text
    """
    language_qualified, lang = language_qualifier(text, classifier)
    return (text_length(text) and language_qualified, lang)
