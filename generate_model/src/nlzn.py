from bs4 import BeautifulSoup
import re
import classla
from conllu import parse
import ast

nlp = None

def word_tokenize(text):
    
    text = clean_text(text)
    doc = nlp(text)
    tokenlist = parse(doc.to_conll())
    return tokenlist_to_list(tokenlist)

def tokenlist_to_list(tokenlist):
    str_token_list = []
    for i in range(len(tokenlist)):
        for j in range(len(tokenlist[i])):
            token = tokenlist[i][j]
            str_token = repr(token)
            str_token_dict = ast.literal_eval(str_token)
            str_token_list.append(str_token_dict['lemma'])
            
    return str_token_list

def clean_text(text):
    # remove html tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # remove special characters and numbers
    pattern = r'[!=\-;:/+,*)@#%(&$_?.^]'
    text =  re.sub(pattern, ' ', text)
    
    # remove backlash
    text.replace("\\", "")
    
    # remove extra white spaces and tabs
    pattern = r'^\s*|\s\s*'
    text =  re.sub(pattern, ' ', text).strip()
    
    return text
        
def setup(language):
    classla.download(lang=language)
    nlp = classla.Pipeline(language, processors='tokenize,lemma')
    
    