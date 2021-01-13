def api_request(request):
    from google.cloud import storage
    import pickle as pk
    from flask import jsonify
    from io import open
    import unicodedata
    import string
    import re
    import random
    import pickle
    import torch
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    def tensorFromSentence(lang, sentence):
         indexes = [lang[word] for word in sentence.split(' ')]
         indexes.append(EOS_token)
         return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    
    def evaluate_eng(encoder, decoder, sentence, max_length=15):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang_kan, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang_eng[topi.item()])

            decoder_input = topi.squeeze().detach()
            sen = ' '.join(decoded_words)

        return sen

    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size
        
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)

        def forward(self, input, hidden):
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
  

    class DecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size):
             super(DecoderRNN, self).__init__()
             self.hidden_size = hidden_size

             self.embedding = nn.Embedding(output_size, hidden_size)
             self.gru = nn.GRU(hidden_size, hidden_size)
             self.out = nn.Linear(hidden_size, output_size)
             self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden):
             output = self.embedding(input).view(1, 1, -1)
             output = F.relu(output)
             output, hidden = self.gru(output, hidden)
             output = self.softmax(self.out(output[0]))
             return output, hidden

        def initHidden(self):
             return torch.zeros(1, 1, self.hidden_size, device=device)


    class AttnDecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=15):
             super(AttnDecoderRNN, self).__init__()
             self.hidden_size = hidden_size
             self.output_size = output_size
             self.dropout_p = dropout_p
             self.max_length = max_length

             self.embedding = nn.Embedding(self.output_size, self.hidden_size)
             self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
             self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
             self.dropout = nn.Dropout(self.dropout_p)
             self.gru = nn.GRU(self.hidden_size, self.hidden_size)
             self.out = nn.Linear(self.hidden_size, self.output_size)

        def forward(self, input, hidden, encoder_outputs):
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)

            attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)

            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

            output = F.log_softmax(self.out(output[0]), dim=1)
            return output, hidden, attn_weights

        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)    
    
    encoder_eng = EncoderRNN(487, 100).to(device)
    attn_decoder_eng = AttnDecoderRNN(100,496,dropout_p=0.1).to(device)
    
    data = {"success": False}
    params = request.get_json()
    if "G1" in params:
         new_row = params.get("G1")
    
    bucket_name = "machine_translator"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    # download and load the model and pickle files
        
    blob = bucket.blob("input_kan")
    blob.download_to_filename("/tmp/input_kan")
    input_lang_kan = pickle.load(open('/tmp/input_kan', 'rb'))
        
    
    blob = bucket.blob("output_eng")
    blob.download_to_filename("/tmp/output_eng")
    output_lang_eng = pickle.load(open('/tmp/output_eng', 'rb'))
        
    blob = bucket.blob("model_enc_eng.dict")
    blob.download_to_filename("/tmp/model_enc_eng.dict")
    encoder_eng.load_state_dict(torch.load('/tmp/model_enc_eng.dict',map_location=torch.device('cpu')))

    blob = bucket.blob("model_dec_eng.dict")
    blob.download_to_filename("/tmp/model_dec_eng.dict")
    attn_decoder_eng.load_state_dict(torch.load('/tmp/model_dec_eng.dict',map_location=torch.device('cpu')))
 

    output_string = evaluate_eng(encoder_eng, attn_decoder_eng,new_row)
     
    data["success"] = True
    return output_string

