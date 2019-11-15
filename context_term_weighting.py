#!/usr/bin/env python
# coding: utf-8
# Author: Saurav Manchanda, manch043@umn.edu, please consider citing """Manchanda, Saurav, Mohit Sharma, and George Karypis. "Intent Term Weighting in E-commerce Queries." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM, 2019.""" and """Manchanda, Saurav, Mohit Sharma, and George Karypis. "Intent term selection and refinement in e-commerce queries." arXiv preprint arXiv:1908.08564 (2019)."""

# In[ ]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


# In[ ]:


import os
import subprocess
import numpy as np
import random
import math


# In[ ]:


DEBUG=False;

def usage():
    return """python context_term_weighting.py  --data_folder <data folder location> 
                                                --embedding_size <Size of the input word embeddings (300 by default)>
                                                --hidden_size_gru <Number of nodes in the hidden layer of GRU (Default 256)>
                                                --hidden_size_mlp <Number of nodes in the hidden layer of MLP (Default 10)>
                                                --dropout <Dropout (Default 0.25)> --num_epochs <Number of training epochs (Default 20)>
                                                --batch_size <Batch size (Default 512)> 
                                                --num_layers_gru <Number of layers in GRU (Default 2)>
                                                --num_layers_mlp <Number of layers in MLP (Default 2)>
                                                --learning_rate <Learning rate (Default 0.001)> --weight_decay <L2 regularization (Default 1e-5)>
                                                --use_cuda <Cuda device to use, negative for cpu (Default -1)> 
                                                --update_embed <Whether to train the embeddings (Default 1)>
                                                --pretrained <Whether to use pretrained embeddings (Default 1; vectors.txt file should be present for this option to work)>
                                                --max_grad_norm <Maximum norm of the gradient, for gradient clipping (Default 1.0)>
                                                --output_file <Path to the output file, to write importance weights>
        """;

def get_args():
    parser = argparse.ArgumentParser(usage=usage());
    parser.add_argument('--data_folder', type=str, default=None, required = True, help='Path to the folder with data.');
    parser.add_argument('--embedding_size', type=int, default=300, required =False, help='Size of the input word embeddings (300 by default).');
    parser.add_argument('--hidden_size_gru', type=int, default=256, required =False, help='Number of nodes in the hidden layer of GRU(Default 256).');
    parser.add_argument('--hidden_size_mlp', type=int, default=10, required =False, help='Number of nodes in the hidden layer of MLP(Default 10).');
    parser.add_argument('--dropout', type=float, default=0.25, required =False, help='Dropout (Default 0.25).');
    parser.add_argument('--num_epochs', type=int, default=10, required =False, help='Number of training epochs (Default 20).');
    parser.add_argument('--batch_size', type=int, default=512, required =False, help='Batch size (Default 512).');
    parser.add_argument('--num_layers_gru', type=int, default=2, required =False, help='Number of layers in GRU (Default 2).');
    parser.add_argument('--num_layers_mlp', type=int, default=2, required =False, help='Number of layers in MLP (Default 2).');
    parser.add_argument('--learning_rate', type=float, default=0.001, required =False, help='Learning rate (Default 0.001).');
    parser.add_argument('--weight_decay', type=float, default=1e-5, required =False, help='L2 regularization (Default 1e-5).');
    parser.add_argument('--use_cuda', type=int, default=0, required =False, help='Cuda device to use, negative for cpu (Default -1).');
    parser.add_argument('--update_embed', type=int, default=1, required =False, help='Whether to train the embeddings (Default 1).');
    parser.add_argument('--pretrained', type=int, default=1, required =False, help='Whether to use pretrained embeddings (Default 1; vectors.txt file should be present for this option to work).');
    parser.add_argument('--max_grad_norm', type=float, default=1.0, required =False, help='Maximum norm of the gradient, for gradient clipping (Default 1.0)');
    parser.add_argument('--output_file', type=str, default=None, required = True, help='Path to the output file, to write importance weights.');
    return parser.parse_args();


# In[ ]:


if DEBUG is True:
    data_folder = "/export/scratch/saurav/trash/split/"
    
    #Optional hyperparameters
    embedding_size = 300;
    hidden_size_gru = 256;
    hidden_size_mlp = 10;
    dropout = 0.25;
    num_epochs = 20;
    batch_size = 512;
    num_layers_gru=2;
    num_layers_mlp=2;
    learning_rate = 0.001;
    weight_decay = 1e-5 # L2 weight regularization
    use_cuda = 0;
    seed = 0;
    pretrained=1;
    update_embed=1;
    max_grad_norm=1.0;
    output_file = "/export/scratch/saurav/trash/lol";
else:
    args = get_args();
    data_folder = args.data_folder;
    embedding_size = args.embedding_size;
    hidden_size_gru = args.hidden_size_gru;
    hidden_size_mlp = args.hidden_size_mlp;
    dropout = args.dropout;
    num_epochs = args.num_epochs;
    batch_size = args.batch_size;
    num_layers_gru = args.num_layers_gru;
    num_layers_mlp = args.num_layers_mlp;
    learning_rate = args.learning_rate;
    weight_decay = args.weight_decay; # L2 weight regularization
    use_cuda = args.cuda;
    seed = args.seed;
    pretrained = args.pretrained;
    update_embed = args.update_embed;
    max_grad_norm = args.max_grad_norm;
    output_file = args.output_file;


# In[ ]:


print("The arguments provided are as follows: ");
print("--data_folder "+str(data_folder));
print("--embedding_size "+str(embedding_size));
print("--hidden_size_gru "+str(hidden_size_gru));
print("--hidden_size_mlp "+str(hidden_size_mlp));
print("--dropout "+str(dropout));
print("--num_epochs "+str(num_epochs));
print("--batch_size "+str(batch_size));
print("--num_layers_gru "+str(num_layers_gru));
print("--num_layers_mlp "+str(num_layers_mlp));
print("--learning_rate "+str(learning_rate));
print("--weight_decay "+str(weight_decay));
print("--use_cuda "+str(use_cuda));
print("--seed "+str(seed));
print("--pretrained "+str(pretrained));
print("--update_embed "+str(update_embed));
print("--max_grad_norm "+str(max_grad_norm));
print("--output_file "+str(output_file));


# In[ ]:


#Set seeds
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True


# In[ ]:


#set computation device
check_cuda = torch.cuda.is_available()
if use_cuda < 0:
    check_cuda = False;
device = torch.device("cuda:"+str(use_cuda) if check_cuda else "cpu")
print("Using device", device)


# In[ ]:


print_every_step = 100;
EPS = 1e-12
test_out = "/export/scratch/saurav/trash/"


# In[ ]:


class Dataset():
    def __init__(self, q1_file, q2_file, vocab_file):
        
        print('='*100)
        print('Dataset preprocessing log:')
        
        print('- Loading vocabulary...')
        self.word_list, self.vocab = self.load_vocab(vocab_file);        
        print('- Vocabulary size: {}'.format(len(self.vocab)))
        
        print('- Loading queries')
        self.q1 = self.load_queries(q1_file, self.vocab)
        self.q2 = self.load_queries(q2_file, self.vocab)
        print('- Loaded '+str(len(self.q1))+" query-pairs")
        
        print('- Counting word frequency')
        self.freq_list = self.build_frequency(len(self.word_list), self.q2);
        self.class_weight = [0.0 for i in range(len(self.freq_list))];
        self.max_occur = float(max(self.freq_list));
        for i in range(len(self.freq_list)):
            if self.freq_list[i] > 0:
                self.class_weight[i] = self.max_occur/self.freq_list[i];
                        
        print('- Self weight labeling')
        self.q1_label = self.annotate_q1(self.q1, self.q2);
        
    def load_vocab(self, vocab_file):
        word_list = []
        vocab = {}
        cnt = 0;
        with open(vocab_file) as f:
            for line in f:
                wrd = line.strip();
                word_list.append(wrd);
                vocab[wrd] = cnt;
                cnt += 1;
        return word_list, vocab;
        
    def load_queries(self, file_path, vocab):
        q_list = []
        with open(file_path) as f:
            for line in f:
                q = []
                q.extend([int(i) for i in line.strip().split()]);
                q_list.append(q);
        return q_list;
    
    def build_frequency(self, word_count, q_list):
        freq_list = [0 for i in range(word_count)];
        for q in q_list:
            for wrd in q:
                freq_list[wrd] += 1;
            
        return freq_list;
    
    
    def annotate_q1(self, q1_list, q2_list):
        q1_label_list = [];
        for i in range(len(q1_list)):
            q1_label = [0 for j in range(len(q1_list[i]))];
            for j in range(len(q1_list[i])):
                if q1_list[i][j] in q2_list[i]:
                    q1_label[j] = 1;
            q1_label_list.append(q1_label);
            
        return q1_label_list;
    
    def query2ids(self, query):
        seq = []
        for i in query:
            if i not in self.vocab:
                seq.append(vocab['<UNK>']);
            else:
                seq.append(vocab[i]);
        return seq
    
    
    def ids2query_external(self, query):
        seq = []
        for i in query:
            seq.append(self.word_list[i]);
            
        return seq


# In[ ]:


class BatchLoader():
    def __init__(self, fail_queries, success_queries, success_labels, batch_size):
        self.data = [[] for i in range(len(fail_queries))];
        for i in range(len(fail_queries)):
            self.data[i].append(i);
            self.data[i].append(fail_queries[i]);
            self.data[i].append(success_queries[i]);
            self.data[i].append(success_labels[i]);
        
        self.data = sorted(self.data, key=lambda x: -1*len(x[1]));
            
        self.change_pts = [];
        self.current_len = -1;
        for i in range(len(self.data)):
            if len(self.data[i][1]) != self.current_len:
                self.current_len = len(self.data[i][1]);
                self.change_pts.append(i);
                
        self.change_pts.append(len(self.data[i][1]));
        
        self.batch_size = batch_size;
        self.batch_count = math.ceil(len(self.data)/self.batch_size);
        self.starting_indices = [i*self.batch_size for i in range(self.batch_count)];
        self.current_count = 0;
        self.next_batch = [i for i in range(self.batch_count)];
        random.shuffle(self.next_batch);
        
    def shuffle_slice(self, a, start, stop):
        i = start
        while (i < stop-1):
            idx = random.randrange(i, stop)
            a[i], a[idx] = a[idx], a[i]
            i += 1
    
    def get_next_batch(self):
        if self.current_count == self.batch_count:
            for i in range(len(self.change_pts)-1):
                self.shuffle_slice(self.data, self.change_pts[i], self.change_pts[i+1]);
            random.shuffle(self.next_batch);
            self.current_count = 0;
        batch_fail_query = [];
        batch_fail_query_len = [];
        batch_success_query = [];
        batch_success_query_len = [];
        batch_success_label = [];
        ids = [];
        temp_batch = self.data[self.starting_indices[self.next_batch[self.current_count]]:min([self.starting_indices[self.next_batch[self.current_count]]+self.batch_size, len(self.data)])];
#         print(str(self.current_count)+" "+str(self.starting_indices[self.next_batch[self.current_count]]))
        max_fail_query_len = max([len(i[1]) for i in temp_batch])
        max_success_query_len = max([len(i[2]) for i in temp_batch])
    
#         print(str(max_query_len));
        for i in range(len(temp_batch)):
            padded_fail_query = np.zeros((max_fail_query_len), dtype=np.int);
            padded_fail_query[:len(temp_batch[i][1])] = temp_batch[i][1];
            
            padded_success_query = np.zeros((max_success_query_len), dtype=np.int);
            padded_success_query[:len(temp_batch[i][2])] = temp_batch[i][2];
            
            padded_label = np.zeros((max_fail_query_len), dtype=np.int);
            padded_label[:len(temp_batch[i][3])] = temp_batch[i][3];
            
            ids.append(temp_batch[i][0]);
            batch_fail_query_len.append(len(temp_batch[i][1]));
            batch_fail_query.append(autograd.Variable(torch.LongTensor(padded_fail_query)).unsqueeze(0));
            
            batch_success_query_len.append(len(temp_batch[i][2]));
            batch_success_query.append(autograd.Variable(torch.LongTensor(padded_success_query)).unsqueeze(0));
            
            batch_success_label.append(autograd.Variable(torch.LongTensor(padded_label)).unsqueeze(0));
            
        self.current_count += 1;
        batch_fail_query = torch.cat(batch_fail_query)
        batch_fail_query_len = torch.LongTensor(batch_fail_query_len)
        batch_success_query = torch.cat(batch_success_query)
        batch_success_query_len = torch.LongTensor(batch_success_query_len)
        batch_success_label = torch.cat(batch_success_label)
        return (ids, batch_fail_query_len, batch_fail_query, batch_success_query_len, batch_success_query, batch_success_label); 


# In[ ]:


word_vectors = os.path.join(data_folder, 'vectors.txt')
vocab_file = os.path.join(data_folder, 'vocab.txt')

train_q1_file = os.path.join(data_folder, 'train_q1.txt')
train_q2_file = os.path.join(data_folder, 'train_q2.txt')

valid_q1_file = os.path.join(data_folder, 'valid_q1.txt')
valid_q2_file = os.path.join(data_folder, 'valid_q2.txt')

test_q1_file = os.path.join(data_folder, 'test_q1.txt')
test_q2_file = os.path.join(data_folder, 'test_q2.txt')


# In[ ]:


dataset = Dataset(train_q1_file, train_q2_file, vocab_file)
valid_dataset = Dataset(valid_q1_file, valid_q2_file, vocab_file)
test_dataset = Dataset(test_q1_file, test_q2_file, vocab_file)


# In[ ]:


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long().to(device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = autograd.Variable(seq_range_expand)
    
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand
    
    return mask


# In[ ]:


class query_encoder(nn.Module):
    def __init__(self, embedding=None, hidden_size_gru=None, hidden_size_mlp=None, num_layers_gru=None, num_layers_mlp=None, dropout=None):
        super(query_encoder, self).__init__()
        
        self.dropout = dropout;
        self.dropout_layer = nn.Dropout(dropout)
        self.hidden_size_gru = hidden_size_gru;
        self.hidden_size_mlp = hidden_size_mlp;
        
        self.embedding = embedding
        self.word_vec_size = self.embedding.embedding_dim
        
        self.num_layers_gru = num_layers_gru;
        self.num_layers_mlp = num_layers_mlp;
        
        self.rnn = nn.GRU(input_size=self.word_vec_size, hidden_size=self.hidden_size_gru//2, num_layers=num_layers_gru, dropout=self.dropout, batch_first=True, bidirectional=True);
        if self.num_layers_mlp < 2:
            self.mlp = nn.ModuleList([nn.Linear(self.hidden_size_gru+self.word_vec_size, 1)]);
        else:
            self.mlp = nn.ModuleList([nn.Linear(self.hidden_size_gru+self.word_vec_size, self.hidden_size_mlp)]);
            for i in range(1, self.num_layers_mlp-1):
                self.mlp.append(nn.Linear(self.hidden_size_mlp, self.hidden_size_mlp))
            self.mlp.append(nn.Linear(self.hidden_size_mlp, 1))
        
    def forward(self, queries, query_lens, hidden=None):
        """
        Args:
            - queries: (max_query_len, batch_size)
            - query_lens: (batch_size)
        Returns:
            - outputs: (max_query_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        
        # (batch_size, max_src_len) => (batch_size, max_src_len, word_vec_size)
        emb = self.embedding(queries)
        emb = self.dropout_layer(emb)
        
        # packed_emb:
        # - data: (sum(batch_sizes), word_vec_size)
        # - batch_sizes: list of batch sizes
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, query_lens, batch_first=True)
        
        # rnn(gru) returns:
        # - packed_outputs: shape same as packed_emb
        # - hidden: (num_layers * num_directions, batch_size, hidden_size) 
        packed_outputs, hidden = self.rnn(packed_emb, hidden)
        
        # outputs: (batch_size, max_src_len, hidden_size * num_directions)
        # output_lens == src_lensˇ
        outputs, output_lens =  nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=True)
        
        output_view = outputs.view(outputs.size(0), outputs.size(1), 2, self.hidden_size_gru//2)
        dummy = torch.zeros(output_view.size(0), output_view.size(3)).unsqueeze(1).to(device);
        forward_prev = torch.cat((dummy, output_view[:,:-1,0,:]), 1).unsqueeze(2);
        backward_prev = torch.cat((output_view[:,1:,1,:], dummy), 1).unsqueeze(2);
        prev = torch.cat((forward_prev, backward_prev), 2);

        diff_tensor = output_view - prev;
        diff_tensor = diff_tensor.view(outputs.size(0), outputs.size(1), outputs.size(2))
        
        diff_tensor = torch.cat((diff_tensor, emb), 2)
        
        
        if len(self.mlp) == 1:
            output = self.mlp[0](diff_tensor);
        else:
            temp_out = F.relu(self.dropout_layer(self.mlp[0](diff_tensor)));
            for i in range(1, len(self.mlp)-1):
                temp_out = F.relu(self.dropout_layer(self.mlp[i](temp_out)));
            output = self.mlp[-1](temp_out);
        
        output = output.squeeze(2)
        # relevance_mask: (batch_size, max_src_len)
        relevance_mask = sequence_mask(query_lens)
        
        # Fills elements of tensor with `-float('inf')` where `mask` is 1.
        output.data.masked_fill_(~relevance_mask.data, -float('inf'))
        output = torch.sigmoid(output)
        
        return output;
        
        # (num_layers * num_directions, batch_size, hidden_size) 
        # => (batch_size, num_layers, hidden_size * num_directions)
#         hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2);
            
            
#         # GRU hidden
#         H = hidden[-1].unsqueeze(1)
            
#         relevance_scores = torch.bmm(H, outputs.transpose(1,2))
        
#         # relevance_mask: (batch_size, seq_len=1, max_src_len)
#         relevance_mask = sequence_mask(query_lens).unsqueeze(1)
        
#         # Fills elements of tensor with `-float('inf')` where `mask` is 1.
#         relevance_scores.data.masked_fill_(1 - relevance_mask.data, -float('inf'))
        
#         # relevance_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax` 
#         # => (batch_size, seq_len=1, max_src_len)
        
#         relevance_weights = F.sigmoid(relevance_scores.squeeze(1))
            
        
#         return relevance_weights;


# In[ ]:


def load_pretrained_embedding(vector_file, vocab):
    
    vocab_size = len(vocab)
    vector_dict = {};
    with open(vector_file) as f:
        for line in f:
            p = line.split();
            vector_dict[p[0]] = [float(i) for i in p[1:]];
            
    rep_size = len(vector_dict['for']);
    
    embedding = np.zeros((vocab_size, rep_size))
    
    print('='*100)
    print('Loading pretrained embedding:')
    print('- Vocabulary size: {}'.format(vocab_size))
    print('- Word vector size: {}'.format(rep_size))
    
    for token in vocab:
        embedding[vocab[token],:] = vector_dict[token];
            
    print('='*100 + '\n')
        
    return torch.from_numpy(embedding).float()


# In[ ]:


#Load embeddings and models
embedding = nn.Embedding(len(dataset.vocab), embedding_size, padding_idx=0);
q_encoder = query_encoder(embedding=embedding, hidden_size_gru=hidden_size_gru, hidden_size_mlp=hidden_size_mlp, num_layers_gru=num_layers_gru, num_layers_mlp=num_layers_mlp, dropout=dropout).to(device);
if pretrained:
    pretrained_embeddings = load_pretrained_embedding(word_vectors, dataset.vocab)
    
    q_encoder.embedding.weight.data.copy_(pretrained_embeddings)
    
    q_encoder.embedding.weight.requires_grad = False;
    if update_embed == 1:
        q_encoder.embedding.weight.requires_grad = True;
        
# Class weights to tensor
class_weights = autograd.Variable(torch.LongTensor(dataset.class_weight)).to(device);
        
q_encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, q_encoder.parameters()),lr=learning_rate, amsgrad=True);
    


# In[ ]:


def masked_cross_entropy(logits, target, length):
    
    logits_minus = -torch.add(logits, -1)
    target_minus = -torch.add(target, -1)
    
    logits = torch.add(logits, EPS);
    logits_minus = torch.add(logits_minus, EPS);
    losses = -target.float()*torch.log(logits) - target_minus.float()*torch.log(logits_minus)
    
    
    # mask: (batch, max_len)
    binary_mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    
    # Note: mask need to bed casted to float!
    losses = losses * binary_mask.float()
    loss = losses.sum() / logits.size(0)
    
    # (batch_size,)
    preds = logits.max(1)[1]
    
    # preds: (batch, 1)
    corrects = torch.gather(target, dim=1, index=preds.unsqueeze(1))
    
    return loss, torch.sum(corrects).item();


# In[ ]:


def train(fail_query_len, fail_query, success_query_len, success_query, success_label, q_encoder, q_encoder_optimizer):    
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    
    batch_size = fail_query.size(0)
    
    
    fail_query = fail_query.to(device);
    fail_query_len = fail_query_len.to(device);
    success_query = success_query.to(device);
    success_query_len = success_query_len.to(device);
    success_label = success_label.to(device);
        
    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    q_encoder.train()
        
    # -------------------------------------
    # Forward query encoder
    # -------------------------------------
    relevance_weights = q_encoder(fail_query, fail_query_len);
        
        
    loss, corrects = masked_cross_entropy(
        relevance_weights.contiguous(),
        success_label.contiguous(),
        fail_query_len)
    
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    q_encoder_optimizer.zero_grad()
    
    #--------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()
    
    
    # Clip gradients
    encoder_grad_norm = nn.utils.clip_grad_norm_(q_encoder.parameters(), max_grad_norm);
    
    # Update parameters with optimizers
    q_encoder_optimizer.step()
    
    return loss.item(), corrects


# In[ ]:


def evaluate(fail_query_len, fail_query, success_query_len, success_query, success_label, q_encoder, q_encoder_optimizer):    
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    
    batch_size = fail_query.size(0)
    
    
    fail_query = fail_query.to(device);
    fail_query_len = fail_query_len.to(device);
    success_query = success_query.to(device);
    success_query_len = success_query_len.to(device);
    success_label = success_label.to(device);
        
    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    q_encoder.eval()
        
    # -------------------------------------
    # Forward query encoder
    # -------------------------------------
    relevance_weights = q_encoder(fail_query, fail_query_len);
        
        
    loss, corrects = masked_cross_entropy(
        relevance_weights.contiguous(),
        success_label.contiguous(),
        fail_query_len)
    
    return loss.item(), corrects, relevance_weights


# In[ ]:


def get_gpu_memory_usage(device_id):
    """Get the current gpu usage. """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map[device_id]


# In[ ]:


def test_stats(outfile=None):
    total_loss = 0
    total_correct_words = 0
    total_words = 0
    id_dict = {}
    if outfile is None:
        batch_loader = BatchLoader(valid_dataset.q1, valid_dataset.q2, valid_dataset.q1_label, batch_size);
    else:
        batch_loader = BatchLoader(test_dataset.q1, test_dataset.q2, test_dataset.q1_label, batch_size);
    for batch_idx in range(batch_loader.batch_count):

        ids, fail_query_len, fail_query, success_query_len, success_query, success_label = batch_loader.get_next_batch();
        
        # Evaluate
        loss, corrects, relevance_weights = evaluate(fail_query_len, fail_query, success_query_len, success_query, success_label, q_encoder, q_encoder_optimizer)
        relevance_weights = relevance_weights.data.cpu().numpy()
        
        for i in range(len(ids)):
            id_dict[ids[i]] = relevance_weights[i,:fail_query_len[i]]
            
        # Statistics.
        total_loss += loss*len(ids)
        total_correct_words += corrects
        total_words += len(ids)
        
            
    if outfile is not None:
        with open(outfile, 'w') as target:
            for i in range(len(test_dataset.q1)):
                temp = [str(j) for j in id_dict[i]];
                target.write(" ".join(temp)+"\n")
        
        
    return total_loss/total_words, 100 * (float(total_correct_words) / total_words)


# In[ ]:


batch_loader = BatchLoader(dataset.q1, dataset.q2, dataset.q1_label, batch_size);
temp_new = 0;
global_step = 0
total_loss = 0
total_correct_words = 0
total_correct_instances = 0
total_words = 0
total_instances = 0
prev_gpu_memory_usage = 0

for epoch in range(num_epochs):
    
    for batch_idx in range(batch_loader.batch_count):
        
        ids, fail_query_len, fail_query, success_query_len, success_query, success_label = batch_loader.get_next_batch();
        
        # Train.
        loss, corrects = train(fail_query_len, fail_query, success_query_len, success_query, success_label, q_encoder, q_encoder_optimizer);
        
        global_step += 1
        total_loss += loss*len(ids)
        total_correct_words += corrects
        total_words += len(ids)
        
        # Print statistics
        if global_step % print_every_step == 0:
            curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
            diff_gpu_memory_usage = curr_gpu_memory_usage - prev_gpu_memory_usage
            prev_gpu_memory_usage = curr_gpu_memory_usage
            total_word_accuracy = 100 * (float(total_correct_words) / total_words)
            
            print('='*100)
            print('Training log:')
            print('- Epoch: {}/{}'.format(epoch, num_epochs))
            print('- Global step: {}'.format(global_step))
            print('- Total loss: {}'.format(total_loss/total_words))
            print('- Total word accuracy: {}'.format(total_word_accuracy))
            print('- Current GPU memory usage: {}'.format(curr_gpu_memory_usage))
            print('- Diff GPU memory usage: {}'.format(diff_gpu_memory_usage))
            print('='*100 + '\n')
            
            
            total_loss = 0
            total_correct_words = 0
            total_words = 0
            total_correct_instances = 0
            total_instances = 0
    print("Validation loss, accuracy: ", test_stats())


# In[ ]:


print("Test loss, accuracy: ", test_stats(output_file))


# In[ ]:




