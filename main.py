from tkinter import *
import tkinter
from tkinter import ttk
import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pandas as pd
import fitz
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
import numpy as np



#doc file
def readfiles(path, pdfs):
   os.chdir(path)
#    pdfs = []
   for file in glob.glob("*.pdf"):
       # print(file)
       pdfs.append(file)
pdfs = []
readfiles('./data csdldpt',pdfs)

def dummy_fun(doc):
    return doc
documents = []
#doc pdf
for fname in pdfs:
    doc = fitz.open(fname)
    text = ''
    for page in doc:
        text += page.get_text()
    documents.append(text)

#chuyen pdf sang chu thuong
lower_documents = []
i = 0
for document in documents:
    i+=1
    new_string = document.lower().translate(str.maketrans('', '', string.punctuation))
    lower_documents.append(new_string)

#token doan van ban
list_of_tokens_without_sw = []
i = 0
print("Dang token doan van ban")
for lower_document in lower_documents:
    text_tokens = word_tokenize(lower_document)
    list_of_tokens_without_sw.append(text_tokens)
    i += 1
    print(i, end = ' ')

#tien xu ly
preprocess_token_list = []
i = 0
print("Dang tien xu ly")
for tokens in list_of_tokens_without_sw:
    i+=1
    print(i,end = ' ')
    tmp = []
    for token in tokens:
        if token.isalpha() and len(token) > 1:
            tmp.append(token)
    preprocess_token_list.append(tmp)
filter_preprocess_token_list = []
i = 0
stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['able','about','above','abroad','according','accordingly','across','actually','adj','after','afterwards','again','against','ago','ahead','all','allow','allows','almost','alone','along','alongside','already','also']
stopwords.extend(newStopWords)
print('Dang loai bo stopwords')
for token_list in preprocess_token_list:
    filtered_words = [word for word in token_list if word not in stopwords]
    filter_preprocess_token_list.append(filtered_words)
    i+= 1
    print(i,end = ' ')










#giao dien
window = Tk()
window.title('Searching form')
window.geometry('1000x1000')
#label
lbl = Label(window, text= 'Input searching string:', font=('Arial', 10))
lbl.grid(column=0,row=0)
#o tim kiem
input_txt = tkinter.Text(window,width=100,height=50)
input_txt.grid(column=0,row = 10)

query = ''




#function nut search
def retrieve_input():
    inputValue=input_txt.get("1.0","end-1c")
    print(inputValue)
    querry = inputValue
    new_querry = querry.lower().translate(str.maketrans('', '', string.punctuation))
    #     print(new_string)
    text_token = word_tokenize(new_querry)
    # filtered_querry = [word for word in text_token if word not in stopwords.words('english')]
    filtered_querry = list(text_token)
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)
    filter_preprocess_token_list.append(filtered_querry)
    X = tfidf.fit_transform(filter_preprocess_token_list)
    tfidf_tokens = tfidf.get_feature_names_out()



    cosine_similarities = linear_kernel(X[0:102], X[X.shape[0]-1]).flatten()
    best_idx = np.argmax(cosine_similarities)
    idx_top_k_score = (-cosine_similarities).argsort()[:10]
    best_doc = pdfs[best_idx]
    best_score = cosine_similarities[best_idx]

    if best_score < 0.05:
        print("no good matches")
    else:
        max_doc = documents[np.argmax(cosine_similarities)]
        print(
            f"Best match ({best_score}):\n\n", best_doc[0:200] + "...",
        )
        for i in idx_top_k_score:
            print(
                f"Top {i} match ({cosine_similarities[i]}):\n\n", pdfs[i] + "..." + "\n"
            )




    # root de show data
    root = Tk()

    frm = Frame(root)
    frm.pack(padx=0, pady=0, anchor=NW)

    tv = ttk.Treeview(frm, columns=(1, 2), show='headings')
    tv.grid(row=0, column=0)

    s = ttk.Style()
    s.theme_use('clam')
    s.configure("Treeview", rowheight=40)

    tv.column("# 1", anchor=CENTER, width=30)
    tv.heading("# 1", text="Stt")
    tv.column("# 2", anchor=W, width=970)
    tv.heading("# 2", text=" Ten")
    root.title('New Data')
    root.geometry('1000x500')
    root.resizable(False, False)

    stt = 0
    # for i in pdfs:
    #     stt += 1
    #     tv.insert(parent='', index='end', text='Parent', values=(stt, i))
    for i in idx_top_k_score:
        print(
            f"Top {i} match ({cosine_similarities[i]}):\n\n", pdfs[i] + "..." + "\n"
        )
        tv.insert(parent='', index='end', text='Parent', values=(cosine_similarities[i], pdfs[i]))
    root.mainloop()
# def handle_search_button():
#     query = input_txt.get()
#     lbl.configure(text = query)
search_btn = Button(window,text='Search',command=retrieve_input)
search_btn.grid(column=10, row = 10)

window.mainloop()

