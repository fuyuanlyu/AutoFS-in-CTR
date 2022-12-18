from absl import flags
import sys, os
import time, random, statistics
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, log_loss
from utils.train_help import get_retrain, get_log, get_cuda, get_optimizer, get_stats, get_dataloader

my_seed = 0
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)

FLAGS = flags.FLAGS
flags.DEFINE_integer("gpu", 0, "specify gpu core", lower_bound=-1, upper_bound=7)
flags.DEFINE_string("dataset", "Criteo", "Criteo, Avazu or KDD12")

flags.DEFINE_string("model", "deepfm", "prediction model")
flags.DEFINE_integer("batch_size", 4096, "batch size")
flags.DEFINE_integer("epoch", 20, "epoch for training/pruning")
flags.DEFINE_integer("latent_dim", 16, "latent dimension for embedding table")
flags.DEFINE_list("mlp_dims", [1024, 512, 256], "dimension for each MLP")
flags.DEFINE_float("mlp_dropout", 0.0, "dropout for MLP")
flags.DEFINE_string("optimizer", "adam", "optimizer for training")
flags.DEFINE_float("lr", 1e-4, "model learning rate")
flags.DEFINE_float("wd", 5e-5, "model weight decay")
flags.DEFINE_float("arch_lr", 1e-3, "architecture param learning rate")
flags.DEFINE_integer("cross_layer_num", 3, "cross layer num") # Deep & Cross Network

# AutoDim
flags.DEFINE_string("transform", "linear", "transform method: linear or zero")

# How to save model
flags.DEFINE_integer("debug_mode", 0, "0 for debug mode, 1 for noraml mode")
flags.DEFINE_string("save_path", "save", "Path to save")
flags.DEFINE_string("save_name", "retrain.pth", "Save file name")
flags.DEFINE_string("arch_file", "arch.npy", "Arch file")
FLAGS(sys.argv)

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class Retrainer(object):
    def __init__(self, opt):
        self.loader = get_dataloader(opt["dataset"], opt["data_path"])
        self.save_path = os.path.join(opt["save_path"], opt['dataset'], opt['model'])
        self.save_name = opt["save_name"]
        self.batch_size = opt["batch_size"]
        self.debug_mode = opt["debug_mode"]

        if opt['cuda'] != -1:
            get_cuda(True, 0)
            self.device = torch.device('cuda')
            opt['train']['use_cuda']=True
        else:
            opt['train']['use_cuda'] = False
        with open(os.path.join(self.save_path, opt["arch_file"]), 'rb') as f:
            arch = np.load(f)
        print(arch)
        self.model = get_retrain(opt['train'], arch).to(self.device)

        self.criterion = F.binary_cross_entropy_with_logits
        self.optimizer = get_optimizer(self.model, opt["train"])
        self.logger = get_log()

    def __update(self, label, data):
        self.model.train()
        for opt in self.optimizer:
            opt.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data)
        loss = self.criterion(prob, label.squeeze())
        loss.backward()
        for opt in self.optimizer:
            opt.step()
        return loss.item()

    def __evaluate(self, label, data):
        self.model.eval()
        data, label = data.to(self.device), label.to(self.device)
        prob = self.model.forward(data)
        prob = torch.sigmoid(prob).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        return prob, label

    def eval_one_part(self, name):
        preds, trues = [], []
        for feature,label in self.loader.get_data(name, batch_size=self.batch_size):
            pred, label = self.__evaluate(label, feature)
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = roc_auc_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
        return auc, loss

    def __save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.save_name))

    def train_epoch(self, max_epoch):
        print('-' * 80)
        print('Begin Training ...')
        params, sparsity = self.model.calc_sparsity()
        self.logger.info("[Params {} | Sparsity {}]".format(params, sparsity))
        step_idx = 0
        best_auc = 0.0
        for epoch_idx in range(int(max_epoch)):
            epoch_step = 0
            train_loss = 0.0
            for feature, label in self.loader.get_data("train", batch_size = self.batch_size):
                step_idx += 1
                epoch_step += 1
                update_loss = self.__update(label, feature)
                train_loss += update_loss
            train_loss /= epoch_step
            val_auc, val_loss = self.eval_one_part(name='val')
            test_auc, test_loss = self.eval_one_part(name='test')
            self.logger.info("[Epoch {} | Train Loss:{}]".format(epoch_idx, train_loss))
            self.logger.info("[Epoch {} | Val Loss:{} | Val AUC: {}]".format(epoch_idx, val_loss, val_auc))
            self.logger.info("[Epoch {} | Test Loss:{} | Test AUC: {}]".format(epoch_idx, test_loss, test_auc))

            if best_auc < val_auc:
                best_auc = val_auc
                best_test_auc, best_test_loss = test_auc, test_loss
                if self.debug_mode == 1:
                    self.__save_model()
            else:
                self.logger.info("Early stopped!!!")
                break
        self.logger.info("Most Accurate | AUC: {} | Logloss: {}".format(best_test_auc, best_test_loss))
    
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
        # field_dim = get_stats("../dataset/criteo/stats_2")
        # data = "../dataset/criteo/threshold_2"
    elif FLAGS.dataset == "Avazu":
        field_dim = get_stats("../../datasets/avazu_stats")
        data = "../../datasets/avazu"
        # field_dim = get_stats("../dataset/avazu/stats_2")
        # data = "../dataset/avazu/threshold_2"
    elif FLAGS.dataset == "KDD12":
        field_dim = get_stats("../../datasets/kdd12_stats")
        data = "../../datasets/kdd12"
        # field_dim = get_stats("../dataset/kdd12/stats_10")
        # data = "../dataset/kdd12/threshold_10"
    
    train_opt = {
        "model":FLAGS.model, "optimizer":FLAGS.optimizer, 
        "lr":FLAGS.lr, "wd":FLAGS.wd, "arch_lr":FLAGS.arch_lr,
        "field_dim": field_dim, "latent_dim":FLAGS.latent_dim, 
        "mlp_dims":FLAGS.mlp_dims, "mlp_dropout":FLAGS.mlp_dropout,
        "cross_layer_num":FLAGS.cross_layer_num, "transform":FLAGS.transform
    }
    opt = {
        "dataset":FLAGS.dataset, "data_path":data, 
        "cuda":FLAGS.gpu, "model":FLAGS.model, "batch_size":FLAGS.batch_size, 
        "save_path":FLAGS.save_path, "save_name":FLAGS.save_name, "debug_mode":FLAGS.debug_mode,
        "arch_file":FLAGS.arch_file, "train":train_opt
    }
    # print("opt:{}".format(opt))

    rter = Retrainer(opt)
    # rter.train_epoch(FLAGS.epoch)
    rter.test_time()

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
