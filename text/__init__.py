""" from https://github.com/keithito/tacotron """
import re

from typing import List

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
  return tokenizer.decode(sequence)