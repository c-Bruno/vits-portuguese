""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from phonemizer import phonemize
import phonemizer


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# Load phonemizer
backend_en_us = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True) # Aqui vamos carregar o phonemizer para en-us
backend_pt_br = phonemizer.backend.EspeakBackend(language='pt-br', preserve_punctuation=True, with_stress=True) # Aqui vamos carregar o phonemizer para pt-br

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  '''Remove accents and special characters'''
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = backend_en_us.phonemize(text, strip=True)
  #phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = backend_en_us.phonemize(text, strip=True)
  #phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def portuguese_cleaners(text):
  """
  Pipeline for Portuguese text.

  This function applies the following steps to the input text:
  1. Removes accents and special characters.
  2. Converts the text to lowercase.
  3. Expands abbreviations.
  4. Uses the 'espeak' backend of the phonemize function to convert the text into phonemes.
  5. Collapses consecutive whitespace characters into a single space.

  Note:
    The 'convert_to_ascii' step and the 'with_stress' argument of the 'phonemize' function are currently commented out.
    If you want to include these steps, you can uncomment them.
  """
  # Remove accents and special characters
  #text = convert_to_ascii(text)

  # Convert the text to lowercase for 
  text = lowercase(text)

  # Expand abbreviations
  #text = expand_abbreviations(text)

  # Convert the text into phonemes using the 'espeak' backend
  phonemes = backend_pt_br.phonemize(text, strip=True)

  #phonemes = phonemize(text, language='pt-br', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)

  # Collapse consecutive whitespace characters into a single space
  phonemes = collapse_whitespace(phonemes)

  return phonemes
