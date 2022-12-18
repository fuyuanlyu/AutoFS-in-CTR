from absl import flags
import sys, os
import time, random, statistics
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, log_loss
from utils.train_help import get_model, get_log, get_cuda, get_optimizer, get_stats, get_dataloader

my_seed = 0
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu", 0, "specify gpu core", lower_bound=-1, upper_bound=7)
flags.DEFINE_string("dataset", "Criteo", "Criteo, Avazu or KDD12")

# General Model
flags.DEFINE_string("model", "deepfm", "prediction model")
flags.DEFINE_integer("batch_size", 4096, "batch size")
flags.DEFINE_integer("epoch", 20, "epoch for training/pruning")
flags.DEFINE_integer("pretrain", 2, "epoch for pretraining")
flags.DEFINE_integer("latent_dim", 16, "latent dimension for embedding table")
flags.DEFINE_list("mlp_dims", [1024, 512, 256], "dimension for each MLP")
flags.DEFINE_float("mlp_dropout", 0.0, "dropout for MLP")
flags.DEFINE_string("optimizer", "adam", "optimizer for training")
flags.DEFINE_float("lr", 1e-4, "model learning rate")
flags.DEFINE_float("wd", 5e-5, "model weight decay")
flags.DEFINE_float("arch_lr", 1e-3, "architecture param learning rate")
flags.DEFINE_integer("cross_layer_num", 3, "cross layer num") # Deep & Cross Network
FLAGS(sys.argv)

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class Trainer(object):
    def __init__(self, opt):
        self.loader = get_dataloader(opt["dataset"], opt["data_path"])
        self.batch_size = opt["batch_size"]

        if opt['cuda'] != -1:
            get_cuda(True, 0)
            self.device = torch.device('cuda')
            opt['train']['use_cuda']=True
        else:
            self.device = torch.device('cpu')
            opt['train']['use_cuda'] = False
        self.model = get_model(opt['train']).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['train']['lr'], weight_decay=opt['train']['wd'])

        # initialize controller
        layer = list()
        layer.append(torch.nn.Linear(opt['train']["latent_dim"], 1))
        layer.append(torch.nn.Softmax(dim=1))
        self.controller = torch.nn.Sequential(*layer).to(self.device)
        self.controller_optimizer = torch.optim.Adam(params=self.controller.parameters(), lr=opt['train']['lr'])

        self.proxy_model = get_model(opt['train']).to(self.device)
        self.proxy_optimizer = torch.optim.SGD(params=self.proxy_model.parameters(), lr=opt['train']['lr'])
        
        self.criterion = F.binary_cross_entropy_with_logits
        self.logger = get_log()    

    def __pretrain(self, label, data):
        data, label = data.to(self.device), label.to(self.device)

        # Update Model
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model.forward(data)
        loss = self.criterion(logits, label.squeeze())
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def __train(self, label, data):
        data, label = data.to(self.device), label.to(self.device)
        
        # Copy to proxy model
        for x, y in zip(self.proxy_model.parameters(), self.model.parameters()):
            x.data.copy_(y.data)

        # Update Model
        self.model.train()
        self.optimizer.zero_grad()
        in_logits = self.model.forward(data, self.controller)
        in_loss = self.criterion(in_logits, label.squeeze())
        in_loss.backward()
        self.optimizer.step()

        # Compute Proxy
        self.proxy_model.train()
        self.proxy_optimizer.zero_grad()
        proxy_logits = self.proxy_model.forward(data, self.controller)
        proxy_loss = self.criterion(proxy_logits, label.squeeze())
        proxy_loss.backward()
        self.proxy_optimizer.step()

        # Update Controller
        self.controller_optimizer.zero_grad()
        out_logits = self.proxy_model.forward(data, self.controller)
        out_loss = self.criterion(out_logits, label.squeeze())
        out_loss.backward()
        self.controller_optimizer.step()

        return in_loss.item()

    def __evaluate(self, label, data, controller=None):
        self.model.eval()
        data, label = data.to(self.device), label.to(self.device)
        if controller is None:
            prob = self.model.forward(data)
        else:
            prob = self.model.forward(data, controller)
        prob = torch.sigmoid(prob).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        return prob, label

    def eval_one_part(self, name, controller=None):
        preds, trues = [], []
        for feature,label in self.loader.get_data(name, batch_size=self.batch_size):
            pred, label = self.__evaluate(label, feature, controller)
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = roc_auc_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
        return auc, loss

    def pretrain(self, max_epoch):
        step_idx = 0
        # Pre-training
        print('-' * 80)
        print('Begin Pretraining ...')
        for epoch_idx in range(max_epoch):
            epoch_step = 0
            train_loss = 0.0
            for feature, label in self.loader.get_data("train", batch_size = self.batch_size):
                step_idx += 1
                epoch_step += 1
                update_loss = self.__pretrain(label, feature)
                train_loss += update_loss
            train_loss /= epoch_step
            val_auc, val_loss = self.eval_one_part('val')
            test_auc, test_loss = self.eval_one_part('test')
            self.logger.info("[Epoch {epoch:d} | Train Loss:{loss:.6f}]".format(epoch=epoch_idx, loss=train_loss))
            self.logger.info("[Epoch {epoch:d} | Val Loss:{loss:.6f} | Val AUC:{auc:.6f}]".format(epoch=epoch_idx, loss=val_loss, auc=val_auc))
            self.logger.info("[Epoch {epoch:d} | Test Loss:{loss:.6f} | Test AUC:{auc:.6f}]".format(epoch=epoch_idx, loss=test_loss, auc=test_auc))

    def train(self, max_epoch):
        step_idx = 0
        best_auc = 0.0
        # Training
        print('-' * 80)
        print('Begin Training ...')
        for epoch_idx in range(int(max_epoch)):
            epoch_step = 0
            train_loss = 0.0
            for feature, label in self.loader.get_data("train", batch_size = self.batch_size):
                step_idx += 1
                epoch_step += 1
                update_loss = self.__train(label, feature)
                train_loss += update_loss
            train_loss /= epoch_step
            val_auc, val_loss = self.eval_one_part('val', self.controller)
            test_auc, test_loss = self.eval_one_part('test', self.controller)
            self.logger.info("[Epoch {epoch:d} | Train Loss:{loss:.6f}]".format(epoch=epoch_idx, loss=train_loss))
            self.logger.info("[Epoch {epoch:d} | Val Loss:{loss:.6f} | Val AUC:{auc:.6f}]".format(epoch=epoch_idx, loss=val_loss, auc=val_auc))
            self.logger.info("[Epoch {epoch:d} | Test Loss:{loss:.6f} | Test AUC:{auc:.6f}]".format(epoch=epoch_idx, loss=test_loss, auc=test_auc))
             
            if best_auc < val_auc:
                best_auc = val_auc
                best_test_auc, best_test_logloss = test_auc, test_loss
            else:
                self.logger.info("Early stopped!!!")
                break
        self.logger.info("Most Accurate | AUC:{auc:.6f} | Logloss:{logloss:.6f}".format(auc=best_test_auc, logloss=best_test_logloss))

    def test_one_time(self):
        mytime = []
        preds, trues = [], []
        index = 0
        for feature, label in self.loader.get_data("val", batch_size=self.batch_size):
            starttime = time.time()
            pred, label = self.__evaluate(label, feature)
            endtime = time.time()
            preds.append(pred)
            trues.append(label)
            mytime.append(endtime - starttime)
            index += 1
        return (sum(mytime) * 1000 / index)

    def test_time(self):
        testtimes = []
        for i in range(5):
            testtime = self.test_one_time()
            testtimes.append(testtime)
        print("Mean: {mean:.6f}".format(mean=statistics.mean(testtimes)))
        print("Std: {std:.6f}".format(std=statistics.stdev(testtimes)))

def main():
    sys.path.extend(["./models","./dataloader","./utils"])
    if FLAGS.dataset == "Criteo":
        field_dim = get_stats("../../datasets/criteo_stats")
        data = "../../datasets/criteo"
    elif FLAGS.dataset == "Avazu":
        field_dim = get_stats("../../datasets/avazu_stats")
        data = "../../datasets/avazu"
    elif FLAGS.dataset == "KDD12":
        field_dim = get_stats("../../datasets/kdd12_stats")
        data = "../../datasets/kdd12"
    
    train_opt = {
        "model":FLAGS.model, "optimizer":FLAGS.optimizer, 
        "lr":FLAGS.lr, "wd":FLAGS.wd, "arch_lr":FLAGS.arch_lr, 
        "field_dim":field_dim, "latent_dim":FLAGS.latent_dim, 
        "mlp_dims":FLAGS.mlp_dims, "mlp_dropout":FLAGS.mlp_dropout,
        "cross_layer_num":FLAGS.cross_layer_num
    }
    opt = {
        "dataset":FLAGS.dataset, "data_path":data, "cuda":FLAGS.gpu, "model":FLAGS.model, 
        "batch_size":FLAGS.batch_size, "train":train_opt
    }
    # print("opt:{}".format(opt))

    trainer = Trainer(opt)
    trainer.test_time()
    # trainer.pretrain(FLAGS.pretrain)
    # trainer.train(FLAGS.epoch)

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
