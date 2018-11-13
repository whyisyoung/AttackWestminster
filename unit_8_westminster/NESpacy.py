import spacy
from collections import Counter
from collections import defaultdict as dd
from multiprocessing import Pool
from spacy.matcher import Matcher
from word2number import w2n
from num2words import num2words as n2w

STOP_VERBS = {'be', "'", 'come', 'would', 'believe', 'take',
    'need', 'say', 'have', 'do', 'will', 'tell', }

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
# nlp = spacy.load('en')

ents = dd(list)
def handle_ent_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    for ent in doc[start:end].ents:
        # print(string_id, ',', ent.text, ',', doc[start:end].text)
        ents[string_id].append(ent.text)

def handle_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    ents[string_id].append(doc[start:end].text)
    # print(string_id, ':', doc[start:end].text)

def handle_verb_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    verbs = ''
    for token in doc[start:end]:
        if token.pos_ == 'VERB':
            if token.lemma_ not in STOP_VERBS:
                verbs += token.lemma_
    if verbs != '':
        ents[string_id].append(verbs)
        # print(string_id, ':', doc[start:end].text, ',', verbs)

def handle_num_match(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    string_id = nlp.vocab.strings[match_id]
    for ent in doc[start:end].ents:
        try:
            numWord = wordAsNum(ent.text)
            ents[string_id].append(numWord)
            # print(string_id, ',', numWord, ',', ent.text, ',', doc[start:end].text)
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
# matcher.add('TypeOfAttack', handle_match,
#     [{}, {'LEMMA': 'attack'}],
#     [{}, {}, {'LEMMA': 'attack'}],
#     [{}, {'LEMMA': 'massacre'}],
#     [{}, {}, {'LEMMA': 'massacre'}],
#     [{}, {'LEMMA': 'shooting'}],
#     [{}, {}, {'LEMMA': 'shooting'}],
#     [{}, {'LEMMA': 'bombing'}],
#     [{}, {}, {'LEMMA': 'bombing'}])
matcher.add('TypeOfAttack', handle_match,
    [{'LEMMA': 'terrorist'}, {'LEMMA': 'attack'}],
    [{'LEMMA': 'bombing'}],
    [{'LEMMA': 'shooting'}])
# matcher.add('PersonDid', handle_verb_match,
#     [{'ENT_TYPE': 'PERSON', 'OP': '+'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'VERB', 'OP': '+'}],
#     [{'LEMMA': 'terrorist'}, {'POS': 'VERB', 'OP': '+'}],
#     [{'LEMMA': 'attacker'}, {'POS': 'VERB', 'OP': '+'}])

f = open('cleaned_text.txt', 'r')
texts = [line for line in f]
def match_text(text):
    doc = nlp(text)
    matcher(doc)
    # return ents

# p = Pool()
# p.map(match_text, texts)

for text in texts:
    doc = nlp(text)
    matcher(doc)
  # for ent in doc.ents:
  #   ents[ent.label_].append(ent.lemma_)
for k, v in ents.items():
  common = Counter(v).most_common(10)
  print(k, ':', common)
