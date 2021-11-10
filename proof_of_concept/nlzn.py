"""
@inproceedings{ljubesic-dobrovoljc-2019-neural,
    title = "What does Neural Bring? Analysing Improvements in Morphosyntactic Annotation and Lemmatisation of {S}lovenian, {C}roatian and {S}erbian",
    author = "Ljube{\v{s}}i{\'c}, Nikola  and
      Dobrovoljc, Kaja",
    booktitle = "Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-3704",
    doi = "10.18653/v1/W19-3704",
    pages = "29--34"
    }
"""
from bs4 import BeautifulSoup
import string
import re
import classla
from conllu import parse
import ast

classla.download(lang='hr')
nlp = classla.Pipeline('hr', processors='tokenize,lemma')

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
        
    
    
    