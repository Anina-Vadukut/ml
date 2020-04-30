import pandas as pd
import re
import spacy
from spacy.matcher import Matcher

TEXTS = [
    "Shampoo with less than 200 rupees", "Gold Massage less than 150 rupees",
    "biscuit"]
    
nlp = spacy.load("en_core_web_sm")
MONEY = nlp.vocab.strings['MONEY']

def add_money_ent(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    doc.ents += ((MONEY, start, end),)
    
def print_results(small):
    for i in small:
        if i: 
            for a in i:
                print(a, end=', ')
                
def main():
    print("Processing %d texts" % len(TEXTS))
    df = pd.read_excel('sampledata.xlsx')
    for text in TEXTS:
        text_split = text.split()
        if len(text_split) < 4:
            for sen in df["Description"]:
                if (all(map(lambda word: word in sen.lower(), text_split))):
                    print(sen)
        doc = nlp(text)
        matcher = Matcher(nlp.vocab)
        matcher.add("MONEY",add_money_ent, [{'LIKE_NUM': True}, {'LOWER': "rupees"}])
        matcher(doc)
        relations = extract_currency_relations(doc)
        for r1, r2, r3 in relations:
            des = r1.orth_.split()            
            if r2.text == 'less than':
                mon = r3.orth_.split()
                for i in mon:
                    if i.isdigit():
                        lis = int(i)
                        d = df.query('Selling_Price <= @lis')
                        small = d["Description"].str.findall('.*?'+'.*'.join(des)+'.*', re.I)
                        print_results(small)

def extract_currency_relations(doc):
    spans = list(doc.ents) 
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    relations = []
    for money in filter(lambda w: w.ent_type_ == "MONEY", doc):
        for i in money.children:
            if money.dep_ in ("appos", "npadvmod", "nsubj"):
                relations.append((money.head, i, money))
            elif money.dep_ in ("pobj", "dobj"):
                relations.append((money.head.head, i, money))
    return relations

if __name__ == "__main__":
    main()
    
