import copy
import numpy as np
import torch
import time
from flcore.clients.clientgp import *
from utils.data_utils import read_client_data
from threading import Thread


class GPFL:
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        args.GCE = GCE(in_features=self.feature_dim,
                            num_classes=args.num_classes,
                            dev=args.device).to(args.device)
        args.CoV = CoV(self.feature_dim).to(args.device)

        self.set_clients(args, clientGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate performance")
                self.evaluate()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            self.global_GCE()
            self.global_CoV()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.feature_extractor)
            
    def global_GCE(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.GCE)

        self.GCE = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.GCE.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_GCE(w, client_model)

        for client in self.clients:
            client.set_GCE(self.GCE)

    def add_GCE(self, w, GCE):
        for server_param, client_param in zip(self.GCE.parameters(), GCE.parameters()):
            server_param.data += client_param.data.clone() * w
            
    def global_CoV(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.CoV)

        self.CoV = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.CoV.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_CoV(w, client_model)

        for client in self.clients:
            client.set_CoV(self.CoV)

    def add_CoV(self, w, CoV):
        for server_param, client_param in zip(self.CoV.parameters(), CoV.parameters()):
            server_param.data += client_param.data.clone() * w


class GCE(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))

        return softmax_loss


class CoV(nn.Module):
    def __init__(self, in_dim):
        super(CoV, self).__init__()

        self.Conditional_gamma = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.Conditional_beta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)

        out = torch.multiply(x, gamma + 1)
        out = torch.add(out, beta)
        out = self.act(out)
        return out