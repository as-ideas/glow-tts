""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import phonemes

class Tokenizer:

  def __init__(self) -> None:
    self.symbol_to_id = {s: i for i, s in enumerate(phonemes)}
    self.id_to_symbol = {i: s for i, s in enumerate(phonemes)}

  def __call__(self, text: str) -> List[int]:
    return [self.symbol_to_id[t] for t in text if t in self.symbol_to_id]

  def decode(self, sequence: List[int]) -> str:
    text = [self.id_to_symbol[s] for s in sequence if s in self.id_to_symbol]
    return ''.join(text)


tokenizer = Tokenizer()


def text_to_sequence(text, cleaner_names, dictionary=None):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
  '''

  return tokenizer(text)


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'
