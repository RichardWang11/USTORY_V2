import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
INPUT_FILE_NAME=''
OUTPUT_FILE_NAME=''
device_id = 0
torch.cuda.set_device(torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'))
print ('Cuda device %s | %s | %s/%sGB' % (torch.cuda.current_device(), torch.cuda.get_device_name(device_id),round(torch.cuda.memory_allocated(device_id)/1024**3,1),round(torch.cuda.memory_reserved(device_id)/1024**3,1)))

#调用jieba库
import jieba

# 中文停用词列表，可以自定义。
STOP_WORDS = set("的了和是就都而及與著或一个".split())

# 这是一个分词函数，使用jieba进行分词
def jieba_tokenizer(doc):
    tokens = jieba.lcut(doc)
    # 对每个token，如果它是字母或数字，并且不在我们的停用词列表中，那么我们进行小写并返回，否则跳过
    return [token.lower() for token in tokens if (token.isalnum() and token not in STOP_WORDS)]
def split_sentence(text):
    return re.split('(?<=[。！？])', text)
  
article_df = pd.read_json(INPUT_FILE_NAME) 
article_df.dropna(subset=['content','title'],inplace=True)
article_df.columns = ['id', 'date', 'title', 'text']
article_df['sentences'] = [[t] for t in article_df.title]
article_df['sentence_counts'] = ""
article_df['sentence_tokens'] = [jieba_tokenizer(t) for t in article_df.title]
#
all_sentences = []
all_sentence_tokens = []
for text in article_df['text'].values:
    sentences = split_sentence(text)
    sentence_tokens = []
    for s in sentences:
        if len(s) > 1:
            tokens = jieba.lcut(s)
            sentence_tokens.append([token.lower() for token in tokens if (token.isalnum() and token not in STOP_WORDS)])
    all_sentences.append(sentences)
    all_sentence_tokens.append(sentence_tokens)
# 将text的句子和对应title的句子拼接，并且把改行的句子数量存到sent counts
for i in range(len(all_sentences)):
    article_df.at[i,'sentences'] = article_df.loc[i].sentences + all_sentences[i]
    article_df.at[i,'sentence_tokens'] = article_df.loc[i].sentence_tokens + all_sentence_tokens[i]
    article_df.at[i,'sentence_counts'] = len(article_df.loc[i].sentences)
    
