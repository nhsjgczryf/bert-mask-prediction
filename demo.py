'''
Input: a sentence with mask tokens.
Output: the sentence in which the mask tokens are replaced by the predictions of PLM.
'''
import transformers
from transformers import AutoModelForMaskedLM,AutoTokenizer
import argparse
import torch
import numpy as np
import copy

transformers.logging.set_verbosity_error()

def search(probs,top_k=1):
    '''
    注意这里我们的search的时候，有一个很好的性质吧，即，比如长为n,且top_k(i,j)代表第i到第j个结点的所有的序列中的概率得分中的第k个概率得分，那么: 
    其实我直觉告诉我，这个性质是这样的，即，top_k对应的序列，其中的每个结点，必定是该位置所有结点概率的前k个中的一个。这个性质有一些类似beam_search,所以我们的搜索也是类似BeamSearch。
    其实很简单的证明，即，采用反证法，即，如果存在一个结点，它不在当前位置的top_k，那么任意一个包含它的序列，它的概率都不会在前面k个。
    很简单:
    对于这样的序列，它的概率得分可以为:
        prob(1,i-1)*prob(i,i)*prob(i+1,n)
    如果有prob(i,i)<top_k(i,i)
    那么有：
        prob(1,i-1)*prob(i,i)*prob(i+1,n)<prob(1,i-1)*top_k(i,i)*prob(i+1,n)<top_k(1,n)
    这就以为着这个序列不可能是我们需要的序列
    args:
        probs: (seq_len,V)
    '''
    top_k_val = []
    top_k_idx = []
    for prob in probs:
        val,idx  = torch.topk(prob,top_k,dim=-1,largest=True,sorted=True)
        top_k_val.append(val)
        top_k_idx.append(idx.tolist())
    #这里我们采用增量搜索的方式，即，我们1到12到123到1234，然后在增量搜索的过程中，我们会drop掉前k个序列之后的序列
    pre_ps = top_k_val[0] #之前的序列的得分
    pre_ids = [[tk] for tk in top_k_idx[0]] #之前的序列（每个元素都是一个列表）
    for i in range(1,len(probs)):
        cur_ps = []
        cur_ids = []
        for pre_p,pre_id in zip(pre_ps, pre_ids):
            for p,_id in zip(top_k_val[i], top_k_idx[i]):
                cur_ps.append(pre_p*p)
                cur_ids.append(pre_id+[_id])
        pre_ps,pre_ids = zip(*sorted(zip(cur_ps,cur_ids),key=lambda x:x[0],reverse=True))
        pre_ps,pre_ids = pre_ps[:top_k],pre_ids[:top_k]
    return pre_ids


def main(sentence_str,args, model,tokenizer):
    input_ids = tokenizer.encode(sentence_str,return_tensors='pt').reshape(1,-1).to(device)
    input_ids1 = input_ids[0].cpu().tolist()[1:-1]
    token_type_ids = torch.zeros(input_ids.shape).to(device)
    position_ids = torch.arange(0,len(input_ids[0])).reshape(1,-1).to(device)+args.offset
    logits = model (input_ids=input_ids.long(),token_type_ids=token_type_ids.long(),position_ids=position_ids.long(),return_dict=True).logits
    logits = logits.squeeze(0)
    logits = logits[1:-1,:] #remove [CLS] and [SEP] (for bert)  or <s> and </s> (for roberta) 
    if args.model_name_or_path.lower().startswith('roberta-'):
        mask = [_id==tokenizer.vocab['<mask>'] for _id in input_ids1]
        mask_idxs =  [i for i,_id in enumerate(input_ids1) if _id==tokenizer.vocab['<mask>']]
    elif args.model_name_or_path.lower().startswith('bert-'):
        mask = [_id==tokenizer.vocab['[MASK]'] for _id in input_ids1]
        mask_idxs =  [i for i,_id in enumerate(input_ids1) if _id==tokenizer.vocab['[MASK]']]
    else:
        raise Exception("Only support BERT and RoBERTa.")
    if args.only_mask:
        probs = torch.softmax(logits[mask,:],dim=-1).cpu().detach()
        ids = search(probs,args.top_k) #(k,num_mask)
        for ids_i in ids:
            t_input_ids = copy.deepcopy(input_ids1)
            for j,_id in enumerate(ids_i):
                t_input_ids[mask_idxs[j]] = _id
            tokens = tokenizer.convert_ids_to_tokens(t_input_ids)
            token_str = tokenizer.convert_tokens_to_string(tokens)
            print(token_str)
    else:
        probs = torch.softmax(logits,dim=-1).cpu().detach()
        ids = search(probs,args.top_k) #(k,seq_len)
        for ids_i in ids:
            tokens = tokenizer.convert_ids_to_tokens(ids_i)
            token_str = tokenizer.convert_tokens_to_string(tokens)
            print(token_str)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str,default='bert-base-uncased')
    parser.add_argument("--top_k",type=int,default=1,help="output tht top k sequences with tht highest probabilities")
    parser.add_argument('--offset',type=int,default=0,help="start position index")
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--only_mask",action="store_true")
    parser.add_argument("--mode",choices=[0,1],type=int,default=1)
    args = parser.parse_args()
    device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu' 
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    while True:
        if args.mode==0:
            sentence_str = input("sentence:")
        elif args.mode==1:
            sentence_str = input("raw sentence:")
            left_mask_num = int(input("left mask number:"))
            right_mask_num = int(input("right mask number:"))
            if args.model_name_or_path.lower().startswith('bert--'):
                sentence_str = '[MASK] '*left_mask_num + sentence_str + ' [MASK]'*right_mask_num
            elif args.model_name_or_path.lower().startswith('roberta-'):
                 sentence_str = '<mask> '*left_mask_num + sentence_str + ' <mask>'*right_mask_num
            else:
                raise Exception()
        else:
            raise Exception()
        main(sentence_str,args,model,tokenizer)
        print('\n')