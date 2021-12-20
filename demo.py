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
    simple beam search.
    args:
        probs: (seq_len,V)
    '''
    top_k_val = []
    top_k_idx = []
    for prob in probs:
        val,idx  = torch.topk(prob,top_k,dim=-1,largest=True,sorted=True)
        top_k_val.append(val)
        top_k_idx.append(idx.tolist())
    pre_ps = top_k_val[0] #previous top k sequence score
    pre_ids = [[tk] for tk in top_k_idx[0]] #previous top k sequence
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
            sentence_str = input("sentence:")
            left_mask_num = int(input("left mask number:"))
            right_mask_num = int(input("right mask number:"))
            if args.model_name_or_path.lower().startswith('bert-'):
                sentence_str = '[MASK] '*left_mask_num + sentence_str + ' [MASK]'*right_mask_num
            elif args.model_name_or_path.lower().startswith('roberta-'):
                 sentence_str = '<mask> '*left_mask_num + sentence_str + ' <mask>'*right_mask_num
            else:
                raise Exception(args.model_name_or_path.lower())
        else:
            raise Exception()
        main(sentence_str,args,model,tokenizer)
