from .transformer import Transformer
from .data_manager import DataManager, Bunch

import io
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

TEXT = """
“I know you haven’t,” said Professor McGonagall, 
sounding half exasperated, half admiring. “But you’re 
different. Everyone knows you’re the only one You- 
Know- oh, all right, Voldemort, was frightened of.” 
“You flatter me,” said Dumbledore calmly. “Voldemort 
had powers I will never have.” 
“Only because you’re too — well — noble to use 
them.” 
“It’s lucky it’s dark. I haven’t blushed so much since 
Madam Pomfrey told me she liked my new earmuffs.”
"""
# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256
# Used for converting between nats and bits
LOG2E = math.log2(math.e)

class NLP:
    def __init__(self):
        # create the model
        arg = Bunch({
            "num_batches":1000000,
            "batch_size":32,
            "lr":0.0001,
            "tb_dir":'./runs',
            "final": True,
            "embedding_size":128,
            "num_heads":8,
            "context":256,
            "depth":12,
            "random_seed":-1,
            "test_every":3000,
            "save_every":1500,
            "test_subset":100000,
            "test_batchsize":64,
            "gradient_clipping":1.0,
            "lr_warmup":5000,
            "wide":True,
            "done_batches":0
        })
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models = {
            'model':Transformer(
                emb=arg.embedding_size,
                heads=arg.num_heads,
                depth=arg.depth,
                seq_length=arg.context,
                num_tokens=NUM_TOKENS,
                wide=arg.wide
            ).to(self.device)
        }
        models['optimizer'] = torch.optim.Adam(lr=arg.lr, params=models['model'].parameters())
            # linear learning rate warmup
        models['test_tensor'] = torch.LongTensor([[]])
        models['scheduler'] = torch.optim.lr_scheduler.LambdaLR(models['optimizer'], lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))
        self.dm = DataManager({**models, **arg}, store_dir='models')

    def sample(self, lnprobs, temperature=1.0):
        """
        Sample an element from a categorical distribution
        :param lnprobs: Outcome log-probabilities
        :param temperature: Sampling temperature. 1.0 follows the given distribution,
            0.0 returns the maximum probability element.
        :return: The index of the sampled element.
        """

        if temperature == 0.0:
            return lnprobs.argmax()

        p = F.softmax(lnprobs / temperature, dim=0)
        cd = dist.Categorical(p)

        return cd.sample()

    @staticmethod
    def text_to_tensor(text):
        return torch.from_numpy(np.fromstring(text, dtype=np.uint8)).to(torch.long)

    def generate(self, seed='', size=1000, temp=0.55):
        context_length = self.dm.context
        seed = TEXT + seed
        seed = seed[len(seed) - context_length:]
        seed_input = NLP.text_to_tensor(seed)
        seed_input = seed_input[len(seed_input) - context_length:]
        print(f'Temperature: {temp}')

        if torch.cuda.is_available():
            seed_input = seed_input.cuda()

        seed_input = Variable(seed_input)

        #old_stdout = sys.stdout
        #new_stdout = io.StringIO()
        #sys.stdout = new_stdout

        out = '['
        for c in seed_input:
            out += str(chr(c))
        out += ']'
        #print(']', end='', flush=True)
        for _ in range(size):
            output = self.dm.model(seed_input[None, :])
            c = self.sample(output[0, -1, :], temp)
            out += str(chr(c))

            seed_input = torch.cat([seed_input[1:], c[None]], dim=0)

        #output = new_stdout.getvalue()
        #sys.stdout = old_stdout

        return str.encode(out).decode('utf-8').replace('â', "'")

    def load(self, path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
        with gzip.open(path) if path.endswith('.gz') else open(path) as file:
            text = file.read()
            text = text.decode('utf-8')
            text = text[:n_train + n_valid + n_test]
            text = text.replace('â',"'")
            X = np.fromstring(text, dtype=np.uint8)
            trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
            return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

    def train(self, data_path):
        data_path = data_path if data_path != '' else '/nlp/data/books_big.txt.gz'
        if self.dm.random_seed < 0:
            seed = random.randint(0, 1000000)
            print('random seed: ', seed)
        else:
            torch.manual_seed(self.dm.random_seed)

        tbw = SummaryWriter(log_dir=self.dm.tb_dir) # Tensorboard logging

        # load the data (validation unless arg.final is true, then test)
        data_train, data_val, data_test = self.load(data_path)
        data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                                if self.dm.final else (data_train, data_val)

        self.dm.test_tensor = data_test.to(torch.long)
        print(self.generate())
        # training loop
        # - note: we don't loop over the data, instead we sample a batch of random subsequences each time.
        num_batches = self.dm.num_batches - self.dm.done_batches
        if self.dm.done_batches > 0:
            print("Restarting from batch:", self.dm.done_batches, " remaining:", num_batches)
        skipped = 0
        for i in tqdm.trange(self.dm.num_batches):
            if skipped <= self.dm.done_batches:
                skipped += 1
            else:
                skipped = float('inf')
                self.dm.optimizer.zero_grad()

                # sample a batch of random subsequences
                starts = torch.randint(size=(self.dm.batch_size, ), low=0, high=data_train.size(0) - self.dm.context - 1)
                seqs_source = [data_train[start  :start + self.dm.context  ] for start in starts]
                seqs_target = [data_train[start+1:start + self.dm.context+1] for start in starts]
                source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
                target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)
                # - target is the same sequence as source, except one character ahead

                if torch.cuda.is_available():
                    source, target = source.cuda(), target.cuda()
                source, target = Variable(source), Variable(target)

                output = self.dm.model(source)

                loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
                tbw.add_scalar('transformer/train-loss', float(loss.item()) * LOG2E, i * self.dm.batch_size)

                loss.backward()

                # clip gradients
                # - If the total gradient vector has a length > 1, we clip it back down to 1.
                if self.dm.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(self.dm.model.parameters(), self.dm.gradient_clipping)

                self.dm.optimizer.step()
                self.dm.scheduler.step()
                self.dm.done_batches += 1
                if i != 0 and (i % self.dm.test_every == 0 or i == num_batches - 1):
                    self.dm.save()
                # - validate every {arg.test_every} steps. First we compute the
                #   compression on the validation (or a subset)
                #   then we generate some random text to monitor progress
                if i != 0 and (i % self.dm.test_every == 0 or i == num_batches - 1):
                    upto = data_test.size(0) if i == num_batches - 1 else self.dm.test_subset
                    data_sub = data_test[:upto]

                    with torch.no_grad():
                        bits, tot = 0.0, 0
                        batch = [] # buffer, every time it fills up, we run it through the model

                        for current in range(data_sub.size(0)):

                            fr = max(0, current - self.dm.context)
                            to = current + 1

                            context = data_sub[fr:to].to(torch.long)
                            if context.size(0) < self.dm.context + 1:
                                pad = torch.zeros(size=(self.dm.context + 1 - context.size(0),), dtype=torch.long)
                                context = torch.cat([pad, context], dim=0)

                                assert context.size(0) == self.dm.context + 1

                            if torch.cuda.is_available():
                                context = context.cuda()

                            batch.append(context[None, :])

                            if len(batch) == self.dm.test_batchsize or current == data_sub.size(0) - 1:

                                # batch is full, run it through the model
                                b = len(batch)

                                all = torch.cat(batch, dim=0)
                                source = all[:, :-1] # input
                                target = all[:, -1]  # target values

                                output = self.dm.model(source)

                                lnprobs = output[torch.arange(b, device=self.device), -1, target]
                                log2probs = lnprobs * LOG2E # convert from nats to bits

                                bits += - log2probs.sum()
                                batch = [] # empty buffer

                        bits_per_byte = bits / data_sub.size(0)

                        # print validation performance. 1 bit per byte is (currently) state of the art.
                        print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
                        tbw.add_scalar(f'transformer/eval-loss', bits_per_byte, i * self.dm.batch_size)

                        print(self.generate())

        self.dm.save()
        print("Done training!")
