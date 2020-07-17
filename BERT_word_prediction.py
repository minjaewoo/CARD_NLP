import sys
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def main():
    txt_path = sys.argv[1]
    with open(txt_path, 'r') as txtfile:
        sentence = txtfile.read().replace('\n', '').replace('.', '')    
    slist = sentence.split(' ')
    output = str()
    
    for i in slist:
        new_sentence_list = ['[CLS]']
        new_sentence_list+=[t if t!=i else '[MASK]' for t in slist]
        new_sentence_list+=['.']
        new_sentence_list+=['[SEP]']
        new_sentence = ' '.join(word for word in new_sentence_list)
        pword = predict_missing_word(new_sentence)+" "
        output+=pword
        print('groundtruth   : %s' %i)
        print('predicted word: %s' %pword)
    print(output)

def predict_missing_word(sentence):
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_index = tokenized_text.index('[MASK]')

    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    return predicted_token



if __name__ == "__main__":
    main()