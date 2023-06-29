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
    
def assign_to_clusters(initial, verbose, window, window_size, to_date, cluster_centers, 
                       cluster_emb_sum_dics, cluster_tf_sum_dics, cluster_topN_probs,
                       T, time_aware = False, theme_aware = False, 
                       cluster_topN_indices = None, cluster_topN_scores = None):
    
    start_time = time.time()
    #根据初始值聚类中心的索引，如果是初始值，则考虑所有聚类中心，否则考虑窗口内文章的聚类中心
    if initial:
        considered_center_indices = list(range(len(cluster_centers)))
    else:
        considered_center_indices = list(set(window[window['cluster']>=0]['cluster']))

    if verbose: print("Assign to "+str(len(considered_center_indices))+" clusters")
    # 计算输出阈值
    out_thred = (1-1/(len(considered_center_indices)+1))**T #+1 to handle a single cluster
    # 考虑主题感知
    if theme_aware:
        sentence_tfs_all = vstack(window[window.cluster==-1]['sentence_TFs'].values)
        article_tfs_all = vstack(window[window.cluster==-1]['article_TF'].values)
        sentence_raw_weights_all = {}
        article_topN_tfs_all = {}
        for cluster_id in considered_center_indices:
            sentence_raw_weights_all[cluster_id] = np.array(np.sum(sentence_tfs_all[:,cluster_topN_indices[cluster_id]].multiply(cluster_topN_scores[cluster_id]), axis=1)).ravel()                       
            article_topN_tfs_all[cluster_id] = article_tfs_all[:,cluster_topN_indices[cluster_id]].toarray()
    # 考虑时间感知        
    if time_aware:
        time_weighted_center_dic = {}
        for uniq_date in window[window.cluster==-1].date.unique():
            for cluster_id in considered_center_indices:
                time_weighted_sum = 0
                time_weighted_num = []
                
                decaying_factor = window_size
                #decaying_factor = len(cluster_emb_sum_dics[cluster_id])
                for date in sorted(cluster_emb_sum_dics[cluster_id].keys())[::-1]: #sorted by newest -> oldest time
                    if (to_date - date).days-1 >= window_size: break #consider only the window context
                    day_delta = np.abs((uniq_date - date).days)
                    time_weighted_num.append(np.exp(-day_delta/decaying_factor)*cluster_emb_sum_dics[cluster_id][date][1]) # time+amount weighted's average
                    # 将时间加权分数和时间嵌入向量相乘（可以将时间加权部分应用于嵌入向量的数值部分），time_weighted_sum给定日期的嵌入向量的加权和
                    time_weighted_sum += np.exp(-day_delta/decaying_factor) * cluster_emb_sum_dics[cluster_id][date][0] 
                # 计算时间加权嵌入向量地平均score
                time_weighted_center = time_weighted_sum/sum(time_weighted_num)
                # 将时间加权向量平均值、待定日期和聚类中心索引一起保存到time_W_c_dic中
                time_weighted_center_dic[(pd.Timestamp(uniq_date), cluster_id)] = time_weighted_center
