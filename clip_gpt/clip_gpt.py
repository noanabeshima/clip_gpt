import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer
import requests
import os

tokenizer = SimpleTokenizer()

def _get_pretrained_model():
    # get directory this file is in
    dir_path = os.path.dirname(os.path.realpath(__file__))
    clip_gpt_fname = dir_path + '/clip_gpt.pt'
    url = 'https://clip-gpt.s3.us-west-1.amazonaws.com/clip_gpt.pt'

    if not os.path.exists(clip_gpt_fname):
        print('Downloading CLIP GPT model from S3...')
        r = requests.get(url, allow_redirects=True)
        open(clip_gpt_fname, 'wb').write(r.content)
        print('Download complete.')

    # load clip model
    clip_gpt_model = torch.load(clip_gpt_fname, map_location='cpu')
    return clip_gpt_model


class CLIPGPT(nn.Module):
    def __init__(self, clip_model=None):
        # initialize from existing OpenAI CLIP model or CLIPGPT model
        super().__init__()
        if clip_model is None:
            clip_model = _get_pretrained_model()
        else:
            assert isinstance(clip_model, torch.nn.Module)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.d_model = self.positional_embedding.shape[-1]
        self.transformer = clip_model.transformer.to(torch.float32)
        self.ln_final = clip_model.ln_final
        # check if deembed is a method of clip_model
        if hasattr(clip_model, 'deembed'):
            # clip model is CLIPGPT
            self.deembed = clip_model.deembed
        else:
            # randomly initialize deembed
            self.deembed = nn.Linear(self.d_model, 49408) # 49408 the size of the clip decoder
        self.start_token = 49406
        self.end_token = 49407

    def forward(self, tokens, return_hidden_states=False):
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding
        
        x = x.permute(1, 0, 2) # BSD -> SBD [batch, seq, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # SBD -> BSD
        
        final_hidden_states = self.ln_final(x)
        if return_hidden_states:
            return final_hidden_states
        else:
            return self.deembed(final_hidden_states)
            

    def generate(self, prompt='', temp=0.7, n_sampled_toks=-1):
        '''
        Generate text from prompt
        n_sampled_toks: number of tokens to sample from the model, -1 for max length (77)
        '''
        global tokenizer
        prompt_tokens = tokenizer.encode(prompt)
        batch = torch.tensor([self.start_token, *prompt_tokens])[None,:].to(self.device)
        if n_sampled_toks == -1:
            n_sampled_toks = 77-batch.shape[-1]
            
        for i in range(batch.shape[-1]-1, min(77-batch.shape[-1], batch.shape[-1]+n_sampled_toks)):
            with torch.no_grad():
                logits = self.forward(F.pad(batch, (0, 77-batch.shape[-1])))[:, i]
            dist = torch.distributions.Categorical(logits=logits/(temp+1e-5))
            sampled_token = dist.sample()
            batch = torch.cat((batch, sampled_token[None]), -1)
            if sampled_token.item() == self.end_token:
                break
        return tokenizer.decode(batch[0][1:-1].tolist())

    @property
    def device(self):
        return self.token_embedding.weight.device
    
