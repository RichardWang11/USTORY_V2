import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
device_id = 0
torch.cuda.set_device(torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'))
print ('Cuda device %s | %s | %s/%sGB' % (torch.cuda.current_device(), torch.cuda.get_device_name(device_id),round(torch.cuda.memory_allocated(device_id)/1024**3,1),round(torch.cuda.memory_reserved(device_id)/1024**3,1)))
#调用spacy库
nlp = spacy.load("en_core_web_lg")
#使用spacy库进行分词，对每篇文档中只返回分词后结果为数字和字母的(al,num)
def spacy_tokenizer(doc):
    tokens = nlp(doc)
    return([token.lemma_.lower() for token in tokens if (token.text.isalnum() and not token.is_stop and not token.is_punct and not token.like_num)])
  
article_df = pd.read_json(INPUT_FILE_NAME) 
# 除去data中text,title列空值的列
article_df.dropna(subset=['text','title'],inplace=True)
# 重命名dataframe的每一列
article_df.columns = ['id', 'date', 'title', 'text', 'story'] # drop story column if not available
# 将dataframe title列中的每个元素用列表[]存储，并放在新的一列sentences
article_df['sentences'] = [[t] for t in article_df.title]
article_df['sentence_counts'] = ""
# 对dataframe title列中每个元素调用spacy分词，放在新的一列
article_df['sentence_tokens'] = [[spacy_tokenizer(t)] for t in article_df.title]
# all sentences存储了所有text中的句子，all sent tokens存了所有句子的分词结果（过滤后）
all_sentences = []
all_sentence_tokens = []
for text in article_df['text'].values:
    parsed = nlp(text)
    sentences = []
    sentence_tokens = []
    for s in parsed.sents:
        if len(s) > 1:
            sentences.append(s.text)
            sentence_tokens.append([token.lemma_.lower() for token in s if (token.text.isalnum() and not token.is_stop and not token.is_punct and not token.like_num)])
    all_sentences.append(sentences)
    all_sentence_tokens.append(sentence_tokens)
# 将text的句子和对应title的句子拼接，并且把改行的句子数量存到sent counts
for i in range(len(all_sentences)):
    article_df.at[i,'sentences'] = article_df.loc[i].sentences + all_sentences[i]
    article_df.at[i,'sentence_tokens'] = article_df.loc[i].sentence_tokens + all_sentence_tokens[i]
    article_df.at[i,'sentence_counts'] = len(article_df.loc[i].sentences)
#https://www.sbert.net/docs/pretrained_models.html
# 加载模型
st_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1').cuda() 
# embeddings存所有df['sentences']中句子编码后的向量，errors是编码错误列表，k用于计算编码的次数
embeddings = []
errors = []
k = 0
for sentences in article_df['sentences']:
    try:
        embedding = st_model.encode(sentences)
        embeddings.append(embedding)
    except Exception as e:
        errors.append(k)
        print("error at", k, e)

    k = k + 1
    if k % 100 ==0:
        print(k)

article_df['sentence_embds'] = embeddings
noise_list = ['hell','shit']
# 遍历dataframe的每一行，是否存在噪声词，噪声词定义在noise list，如果句子中有噪声词，删除这一行
for (idx,row) in article_df.iterrows():
    for n in noise_list:
        if n in row['sentences']:
            article_df.drop(idx, inplace = True)
            break

article_df['date'] = [str(k)[:10] for k in article_df['date']]
article_df.sort_values(by=['date'],inplace=True)
article_df.reset_index(inplace= True, drop=True)
article_df['id'] = article_df.index
article_df.to_json(OUTPUT_FILE_NAME)
