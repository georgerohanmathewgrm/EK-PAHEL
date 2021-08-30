# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:00:26 2021

@author: groha
"""

# -- coding: utf-8 --
"""
Created on Sat Aug 28 16:30:48 2021

@author: delwi
"""

# -- coding: utf-8 --
"""
Created on Thu Aug 12 19:34:18 2021

@author: delwi
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.tokenize import word_tokenize
url = 'https://www.ndtv.com/topic/ngo'
agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
page = requests.get(url, headers=agent)
soup=BeautifulSoup(page.content, 'lxml')
link = []
text1=[]
li=[]
headline=[]
link=[]
description=[]

k=soup.find('div', {'id':'news_list'})


z=0
for i in soup.find('div', {'id':'news_list'}).find_all('li'):
  
    headline.append(i.a.text)
   
    link.append(i.a['href'])
   
    description.append(i.find('div',{'class':'src_itm-txt'}).text)
print(headline[1],link[1],description[1])  
   
"""
    try:
        if(z==0):
            text1.append(i.img['alt'])
            li.append(i['href'])
            z=1
        i['href'] ='https://www.hindustantimes.com/'+i['href'] + '?page=all'
        link.append(i['href'])
    except:
        continue
mydivs = soup.findAll("div", {"class": "cartHolder page-view-candidate listView"})
item=mydivs[0]
for i in mydivs:
     try:
        i.a['href'] =i.a['href'] + '?page=all'
        link.append(i.a['href'])        
        text1.append(i.img['alt'])
        
        li.append(i.a['href'])
     except:
        continue
para=0
documents = []
number=0
cou=0
k=0
for i in link:
    try:
     agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
     page = requests.get(i, headers=agent)
    except:
        link.pop(k)
    k=k+1
for i in link:
    number=number+1
    try:       
        # Make a request to the link
        # Initialize BeautifulSoup object to parse the content 
        agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        page = requests.get(i, headers=agent)
        soup=BeautifulSoup(page.content, 'lxml')
        # Retfl paragraphs and combine it as one
        print("number",number,":",i) 
    except:
        i='https://www.hindustantimes.com/'+i
        link[number-1]=i
        agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        page = requests.get(i, headers=agent)
        soup=BeautifulSoup(page.content, 'lxml')
        print("number",number,":",i)
    sen = []
io=0
while(io==0):    
    number=int(input("number"))
    lii=link[number-1]
    print(lii)
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    page = requests.get(lii, headers=agent)
    soup=BeautifulSoup(page.content, 'lxml')               
    for i in soup.find('div',{'class':'storyDetails'}).find_all('p'):           
                if(i.a==None and i.text !="Get our daily newsletter"):
                    sen.append(i.text)
                    para=para+1                 
                    print(para,":")
                    print(i.text)
    documents.append(' '.join(sen))
    if(documents==['']):
        print("link is invalid,choose another number")
        documents=[]
        io=0
    else:
        io=1
import re
import string
documents_clean = []
for d in documents:
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove punctuations
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    documents_clean.append(document_test)
ham=[]
for d in text1:
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove punctuations
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    ham.append(document_test)
df = pd.DataFrame({"Heading": ham,"Link":li})

df.to_csv('news.csv')
tokens=word_tokenize(documents_clean[0])
noduplitoken=list(dict.fromkeys(tokens))
print(len(tokens))
print("number of paragraphs",para)
print("Number of words (without duplicates):",len(noduplitoken))
wordfreq=[]
for w in tokens:
    wordfreq.append(tokens.count(w))
res=dict(zip(tokens,wordfreq))
print(res)
print("Mostly used word from an entire website ",tokens[wordfreq.index(max(wordfreq))])
"""