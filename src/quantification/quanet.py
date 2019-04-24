import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from quantification.helpers import *
from time import time
from tqdm import tqdm

from util.plot_correction import plot_corr


class QuaNet(torch.nn.Module):

    def __init__(self, inputsize, lstm_hiddensize, lstm_layers, ff_sizes, device, bidirectional=True, drop_p=0.5):
        super().__init__()

        self.lstm_hiddensize = lstm_hiddensize
        self.lstm_layers = lstm_layers

        self.drop_p = drop_p
        self.input_size=inputsize+2 # the dimensionality of the matrix plus the predicted label and probability estimate
        self.device = device
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(self.input_size, lstm_hiddensize, lstm_layers, bidirectional=bidirectional, dropout=drop_p)

        prev_size = self.lstm_hiddensize * (2 if bidirectional else 1)
        stats_size = 8*2 # number of classification stats (cc, acc, ..., tpr, fnr, 1-cc, 1-acc,...,1-fnr)
        prev_size += stats_size
        self.ff = torch.nn.ModuleList()
        for linear_size in ff_sizes:
            self.ff.append(torch.nn.Linear(prev_size, linear_size))
            prev_size = linear_size

        self.output = torch.nn.Linear(prev_size, 1)
        self.dropout = torch.nn.Dropout(drop_p)
        self.to(device)


    def init_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1
        var_hidden = torch.zeros(self.lstm_layers * directions, batch_size, self.lstm_hiddensize).to(self.device)
        var_cell = torch.zeros(self.lstm_layers * directions, batch_size, self.lstm_hiddensize).to(self.device)
        return (var_hidden, var_cell)


    def forward(self, x, stats=None):
        batch_size = x.size()[0]
        rnn_output, _ = self.lstm(x.transpose(0, 1), self.init_hidden(batch_size))
        abstracted = rnn_output[-1]

        if stats is not None:
            # abstracted = torch.cat([stats.view(abstracted.shape[0],-1), abstracted], dim=1)
            abstracted = torch.cat([stats, abstracted], dim=1)

        for linear in self.ff:
            abstracted = self.dropout(F.relu(linear(abstracted)))

        # prevalence = F.softmax(self.output(abstracted), dim=-1)[:,1]
        # prevalence = ((prevalence-0.5)*1.2 + 0.5) # scales the sigmoids so that the net is able to reach either 1 or 0
        # if not self.training:
        #      prevalence = torch.clamp(prevalence, 0, 1)

        # prevalence = torch.sigmoid(self.output(abstracted)).view(-1)
        prevalence = self.output(abstracted).view(-1)
        if not self.training:
             prevalence = torch.clamp(prevalence, 0, 1)

        return prevalence

    def fit(self, X, y, yhat, yprob, lr = 1e-4, wd = 1e-4, batch_size=21 * 5, sample_size=500, maxiterations = 10000,
            patience = 10, validate_every = 10, val_prop=0.33, model_path='quanet.dat'):

        #batch_size=21*5 means "take 5 samples for each prevalence (there are 21 different prevalences in the range)

        #todo: try only with yva, yhatva
        tpr_val = tpr(y, yhat)
        fpr_val = fpr(y, yhat)
        ptpr_val = prob_tpr(y, yprob)
        pfpr_val = prob_fpr(y, yprob)

        Xtr, Xva, ytr, yva, yhattr, yhatva, yprobtr, yprobva = train_test_split(X, y, yhat, yprob, stratify=y, test_size=val_prop)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)

        prevs_range = prevalence_range()
        batch_prevalences = np.repeat(prevs_range, batch_size / prevs_range.size)

        print('Init quant_net training:')
        best_mse, last_mse = -1, -1
        losses = []
        max_patience = patience
        pbar = tqdm(range(1, maxiterations+1))

        sample_indexes_tr = sample_indexes_at_prevalence(ytr, prevalence_range(5), sample_size)
        sample_indexes = sample_indexes_at_prevalence(yva, prevalence_range(5), sample_size)
        sampler = sample_generator(Xtr, ytr, yhattr, yprobtr, batch_prevalences, sample_size)

        # prevs_tails = np.array([0.01,0.05,0.10,0.15,0.85,0.90,0.95,0.99])
        # batch_prevalences_tails = np.repeat(prevs_tails, batch_size / prevs_tails.size)
        # sampler_tails = sample_generator(Xtr, ytr, yhattr, yprobtr, batch_prevalences_tails, sample_size)

        # for the plot
        # _, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs = Baselines_predictions(Xval, yval, yhatval, sample_size, n_test_for_prevalence=5)

        self.train()
        for step in pbar:

            # if step%2==0:
            #     batch_Xyhat, batch_stats, batch_real_prevalences = next(sampler)
            # else:
            #     batch_Xyhat, batch_stats, batch_real_prevalences = next(sampler_tails)

            batch_Xyhat, batch_stats, batch_real_prevalences = next(sampler)

            batch_Xyhat = torch.FloatTensor(batch_Xyhat).to(self.device)
            batch_stats = torch.FloatTensor(batch_stats).to(self.device)
            batch_real_prevalences = torch.FloatTensor(batch_real_prevalences).to(self.device)

            optimizer.zero_grad()
            predicted_prevalences = self.forward(batch_Xyhat, batch_stats)
            quant_loss = criterion(predicted_prevalences, batch_real_prevalences)
            quant_loss.backward()
            optimizer.step()

            loss = quant_loss.item()
            losses.append(loss)

            running_loss, running_loss_std = np.mean(losses[-20:]), np.std(losses[-20:])
            pbar.set_description('loss {:.6f} (+-{:.6f}) patience={} mse(best)={:.6f} mse(last)={:.6f}'.format(
                running_loss, running_loss_std, patience, best_mse, last_mse))

            if step % validate_every == 0:

                true_prevs, pred_prevs_tr, _,_,_,_ = QuaNet_predictions(self, Xtr, ytr, yhattr, yprobtr, tpr_val, fpr_val, ptpr_val, pfpr_val, sample_indexes_tr, batch_size)
                true_prevs, pred_prevs_val, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs = QuaNet_predictions(self, Xva, yva, yhatva, yprobva, tpr_val, fpr_val, ptpr_val, pfpr_val, sample_indexes, batch_size)

                last_mse = mse(true_prevs,pred_prevs_val)

                methods = [pred_prevs_tr, pred_prevs_val, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs]
                labels = ['quanet_tr','quanet_val', 'cc', 'acc', 'pcc', 'apcc']
                plot_corr(true_prevs, methods, labels, savedir='../plots', savename='tmp.pdf', title='steps={} mse={:.5f}'.format(step, last_mse))

                scheduler.step(running_loss)

                # mse_net_sample = mse(true_prevs, net_prevs)
                if best_mse == -1 or last_mse < best_mse:
                    torch.save(self.state_dict(), model_path)
                    best_mse = last_mse
                    patience = max_patience
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stop after {} loss checks without improvement'.format(max_patience))
                        break

        print('restoring best model')
        self.load_state_dict(torch.load(model_path))


    def predict(self, Xyhat, stats):
        mode=self.training
        self.eval()

        if Xyhat.ndim==2:
            Xyhat = np.expand_dims(Xyhat, axis=0)
            stats = np.expand_dims(stats, axis=0)

        Xyhat = torch.FloatTensor(Xyhat).to(self.device)
        stats = torch.FloatTensor(stats).to(self.device)
        predicted_prevalences = self.forward(Xyhat, stats)

        self.train(mode)

        return predicted_prevalences.cpu().detach().numpy()