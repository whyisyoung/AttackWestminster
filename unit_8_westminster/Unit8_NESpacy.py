# Author: Colm Gallagher
# Unit 8
# This script reads in the cleaned_text.txt file which is a plain txt output of
# our cleaned dataset, and uses spaCy's named entity recognizer and pattern
# based matcher to extract values for our Unit 8 to be used in Unit 9's summary.
#
# The script is multiprocessed for better runtime.

import spacy
from collections import Counter
from collections import defaultdict as dd
from multiprocessing import Pool
from spacy.matcher import Matcher
from word2number import w2n
from num2words import num2words as n2w

def isNumber(text):
    try:
        w2n.word_to_num(text)
    except:
        return False
    return True

def numAsWord(text):
    return n2w(w2n.word_to_num(text))

def wordAsNum(text):
    text = text.lower()
    if text == 'dozens':
        return text
    text = text.replace('nearly', '<')
    text = text.replace('more than', '>')
    text = text.replace('at least', '>')
    text = text.replace('about', '~')
    text = text.replace('around', '~')
    text = text.replace('approximately', '~')
    text = text.replace('almost', '<')
    text = text.replace('up to', '<')
    text = text.replace(',', '')
    if text[0] in {'<', '>', '~'}:
        text = text[1:].strip()
    return str(w2n.word_to_num(text))


nlp = spacy.load('en_core_web_md')

ents = dd(list)
def handle_ent_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    for ent in doc[start:end].ents:
        ents[string_id].append(ent.text)

def handle_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    ents[string_id].append(doc[start:end].text)

def handle_num_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    for ent in doc[start:end].ents:
        try:
            numWord = wordAsNum(ent.text)
            ents[string_id].append(numWord)
        except:
            print(ent.text, 'is not a number')

matcher = Matcher(nlp.vocab)
matcher.add('Attacker', handle_ent_match,
    [{'LEMMA': 'attacker'}, {'ENT_TYPE': 'PERSON', 'OP': '+'}],
    [{'LEMMA': 'terrorist'}, {'ENT_TYPE': 'PERSON', 'OP': '+'}],
    [{'ENT_TYPE': 'PERSON', 'OP': '+'}, {'LEMMA': 'attack'}])
matcher.add('Location', handle_ent_match,
    [{'LEMMA': 'in'}, {'ENT_TYPE': 'GPE', 'OP': '+'}],
    [{'LEMMA': 'at'}, {'ENT_TYPE': 'GPE', 'OP': '+'}],
    [{'LEMMA': 'on'}, {'ENT_TYPE': 'GPE', 'OP': '+'}],
    [{'LEMMA': 'by'}, {'ENT_TYPE': 'GPE', 'OP': '+'}])
matcher.add('Near', handle_ent_match,
    [{'LEMMA': 'in'}, {'ENT_TYPE': 'FAC', 'OP': '+'}],
    [{'LEMMA': 'at'}, {'ENT_TYPE': 'FAC', 'OP': '+'}],
    [{'LEMMA': 'on'}, {'ENT_TYPE': 'FAC', 'OP': '+'}],
    [{'LEMMA': 'by'}, {'ENT_TYPE': 'FAC', 'OP': '+'}],
    [{'LEMMA': 'near'}, {'ENT_TYPE': 'FAC', 'OP': '+'}])
matcher.add('Killed', handle_num_match,
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'victim'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'victim'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'victim'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'dead'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'dead'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'dead'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'die'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'die'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'die'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'kill'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'kill'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'kill'}],
    [{'LEMMA': 'kill'}, {'ENT_TYPE': 'CARDINAL', 'OP': '+'}])
matcher.add('Injured', handle_num_match,
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'injure'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'injure'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'injure'}],
    [{'LEMMA': 'injure'}, {'ENT_TYPE': 'CARDINAL', 'OP': '+'}],
    [{'LEMMA': 'injure'}, {}, {'ENT_TYPE': 'CARDINAL', 'OP': '+'}],
    [{'LEMMA': 'injure'}, {}, {}, {'ENT_TYPE': 'CARDINAL', 'OP': '+'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'hurt'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'hurt'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'hurt'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {'LEMMA': 'wound'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {'LEMMA': 'wound'}],
    [{'ENT_TYPE': 'CARDINAL', 'OP': '+'}, {}, {}, {'LEMMA': 'wound'}])
matcher.add('Date', handle_ent_match,
    [{'ENT_TYPE': 'DATE'}, {'ENT_TYPE': 'DATE'}, {'IS_PUNCT': True, 'OP': '?'}, {'ENT_TYPE': 'DATE'}])
matcher.add('TypeOfAttack', handle_match,
    [{'LEMMA': 'terrorist'}, {'LEMMA': 'attack'}],
    [{'LEMMA': 'terrorist'}, {'LEMMA': 'incident'}],
    [{'LEMMA': 'bombing'}],
    [{'LEMMA': 'shooting'}])

f = open('cleaned_text.txt', 'r')
texts = [line for line in f]
f.close()
def match_text(text):
    global ents
    ents = dd(list)
    doc = nlp(text)
    matcher(doc)
    return ents

p = Pool()
map_ents = p.map(match_text, texts)
for map_ent in map_ents:
    for k, v in map_ent.items():
        ents[k].extend(v)

f = open('Unit8Results.txt', 'w+')
for k, v in ents.items():
  common = Counter(v).most_common(10)
  f.write("%s: %s\n" % (k, str(common)))
  print(k, ':', common)
f.close()
