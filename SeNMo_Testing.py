import os
import logging
import numpy as np
import argparse
import math
import pickle5 as pickle
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import accuracy_score
from joblib import load
import warnings
mpl.rcParams['axes.linewidth'] = 3
warnings.filterwarnings('ignore')

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regression', type=str, default=True, help="Task type, for regression:True, for classficaiton: False")
    parser.add_argument('--dataroot', type=str, default='', help="datasets")
    parser.add_argument('--checkpoints_dir', type=str, default='', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv', help='name of the project. It decides where to store samples and models, class for classification, surv for survival analysis')
    parser.add_argument('--model_name', type=str, default='omic', help='omic for combined omic model, mirna for miRNA model, dnamethyl for DNA Methylation model, gene-expr for gene expression model')
    parser.add_argument('--disease', type=str, default='pancancer_indl_cancers', help='type of the data, pancancer_combined | pancancer_individual_Mod | pancancer_indl_cancers | pancancer_cl. It decides where to store models, features and results for each cancer')
    parser.add_argument('--cancer', type=str, default=None, help='cancer type, TCGA-ACC | TCGA-BLCA | TCGA-BRCA | TCGA-CESC | TCGA-CHOL | TCGA-COAD | TCGA-DLBC | TCGA-ESCA | TCGA-GBM | TCGA-HNSC | TCGA-KICH | TCGA-KIRC | TCGA-KIRP | TCGA-LAML | TCGA-LGG | TCGA-LIHC | TCGA-LUAD | TCGA-LUSC | TCGA-MESO | TCGA-OV | TCGA-PAAD | TCGA-PCPG | TCGA-PRAD | TCGA-READ | TCGA-SARC | TCGA-SKCM | TCGA-STAD | TCGA-TGCT | TCGA-THCA | TCGA-THYM | TCGA-UCEC | TCGA-UCS | TCGA-UVM')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0,1,2, use -1 for CPU')
    parser.add_argument('--mode', type=str, default='omic', help='mode')
    parser.add_argument('--task', type=str, default='surv', help='surv | grad | class')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function, Sigmoid | None | LSM, LSM for classification')
    parser.add_argument('--input_size_omic', type=int, default=80697, help="input_size for omic vector, pancancer dnamethyl=52396, pancancer gene-expr=8794, pancancer mirna=1730, pancancer 3modal=62920, pancancer 4modal=63381, pancancer 5modal=80631, pancancer 6modal=80697, CPTAC=19956")
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--reg_type', default='all', type=str, help="regularization type")
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='start of epoch')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--hidden_layers', type=list, default=[1024, 512, 256, 128, 48, 48, 48], help="Number and size of hidden layers. Default: [512, 256, 48, 32, 32]") # pan-cancer: [1024, 512, 256, 128, 48, 48, 48]
    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    parser.add_argument('--lambda_nll', type=float, default=1)
    parser.add_argument('--init_type', type=str, default='max', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    parser.add_argument('--lr', default=2e-3, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--weight_decay', default=4e-4, type=float, help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--patience', default=0.005, type=float)
    opt = parser.parse_known_args()[0]
    printoptions(parser, opt)
    opt = parsegpus(opt)
    return opt

def printoptions(parser, opt):
    """Print default values(if different) and save options to a text file [checkpoints_dir]/opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    mkfolders(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def parsegpus(opt):
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    return opt

def mkfolders(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)

def define_activation(act_type='ReLU'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "None":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

def mixedcollate(batch):  
    transposed = zip(*batch)
    return [default_collate(samples) for samples in transposed]

def initweights(net, init_type='orthogonal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def initmaxweights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

def initnet(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize the network: 1. register CPU/GPU device, 2. initialize network weights"""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'max' and init_type != 'none':
        print("Init Type:", init_type)
        initweights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net

class SeNMo(nn.Module):
    def __init__(self, input_dim=1000, hidden_layers=[512, 256, 48, 48, 48], dropout_rate=0.25, act=None, label_dim=1, init_max=True, regression=True):
        super(SeNMo, self).__init__()
        self.act = act
        self.regression = regression
        layers = []

        # Add dynamic hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout_rate, inplace=False))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        if self.regression:
            self.classifier = nn.Sequential(nn.Linear(prev_dim, label_dim))
        else:
            self.classifier = nn.Sequential(nn.Linear(prev_dim, 33))  # For 33-class classification
        if init_max:
            initmaxweights(self)
        if self.regression:
            self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
            self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        else:
            self.output_range = None
            self.output_shift = None

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.encoder(x)
        out = self.classifier(features)
        if self.regression:
            if self.act is not None:
                out = self.act(out)
                if isinstance(self.act, nn.Sigmoid):
                    out = out * self.output_range + self.output_shift
        else:
            out = self.act(out) # Apply LogSoftmax for classification, opt.act_type = LSM
        return features, out


def definenet(opt):
    net = None
    act = define_activation(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False
    net = SeNMo(input_dim=opt.input_size_omic, hidden_layers=opt.hidden_layers, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=init_max, regression=opt.regression)
    return initnet(net, opt.init_type, opt.init_gain, opt.gpu_ids)

def definereg(opt, model):
    loss_reg = None
    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'all':
        loss_reg = reg_weights(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg

def reg_weights(model):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

class MolecularDatasetLoader(Dataset):
    def __init__(self, opt, data, split):
        """
            X = data
            e = overall survival event (vital status)
            t = overall survival in months
        """
        self.X_patname = data[split]['x_patname']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        # self.g = data[split]['g']
        # self.transforms = transforms.Compose([
        #                     transforms.RandomHorizontalFlip(0.5),
        #                     transforms.RandomVerticalFlip(0.5),
        #                     transforms.RandomCrop(opt.input_size_path),
        #                     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __getitem__(self, index):
        if opt.regression:
            single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        else:
            # assign 0 to single_e if it is not a regression task
            single_e = torch.tensor(0).type(torch.FloatTensor) 
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor) # for regression task this is the survival time, for classification task this is the class label 0-32
        single_X_patname = self.X_patname[index]
        single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
        return (single_X_patname, single_X_omic, single_e, single_t, 0)
    def __len__(self):
        return len(self.X_patname)

def countparameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Evaluation metrics

def CoxLoss(survtime, censor, hazard_pred, device):
    # Credit to http://traversc.github.io/cox-nnet/docs/
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def accuracycox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def cox_logrank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)

def CIndex(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))

### Test / Inference Function
def test(opt, model, data, split, device):
    model.eval()
    custom_data_loader = MolecularDatasetLoader(opt, data, split=split)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=False, collate_fn=mixedcollate, num_workers=8, pin_memory=True, prefetch_factor=3, persistent_workers=True)
    print("Number of %s samples: %d" % (split, len(test_loader.dataset)))
    print("Number of %s batches: %d" % (split, len(test_loader)))
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    patient_all, features_all = np.array([]), []
    probs_all, gt_all = None, np.array([])
    loss_test, class_acc_test = 0, 0

    for batch_idx, (x_patname, x_omic, censor, survtime, grade) in enumerate(test_loader):
        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade
        survtime = survtime.to(device) if "class" in opt.task else survtime
        survtime = survtime.long() if "class" in opt.task else survtime  # Ensure that survtime is of type long
        features, pred = model(x_omic=x_omic.to(device))
        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_reg = definereg(opt, model)
        loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
        loss_class = F.cross_entropy(pred, survtime) if opt.task == "class" else 0  # Classification loss
        if opt.task == "class":
            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*(loss_nll + loss_class) + opt.lambda_reg*loss_reg
        else:
            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
        loss_test += loss.data.item()
        gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))

        if opt.task == "surv":
            patient_all = np.concatenate((patient_all, x_patname))
            # Ensure that Batch Size is 1 so the features are saved as separate arrays
            features_all.append(features.detach().cpu().numpy().reshape(-1))
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        elif opt.task == "class":
            preds = np.argmax(pred.detach().cpu().numpy(), axis=1)
            test_pred_all = np.concatenate((test_pred_all, preds))
            test_true_all = np.concatenate((test_true_all, survtime.detach().cpu().numpy()))
    
    loss_test /= len(test_loader)
    cindex_test = CIndex(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_logrank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracycox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    class_acc_test = accuracy_score(test_true_all, test_pred_all) if opt.task == 'class' else None
    # grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' else None
    if opt.task == "class":
        pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all, test_pred_all, test_true_all]
    else:
        pred_test = [patient_all, risk_pred_all, survtime_all, censor_all, probs_all, gt_all]
    
    pat_features = [patient_all, features_all]
    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, pat_features, class_acc_test
    # return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, pat_features

### Initializes parser and device
opt = parseargs()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease))

### Initializes Data
results_test = []
features = []
if opt.cancer is None:
    data_cv_path_template = os.path.join(opt.dataroot, opt.disease, '{}_combined.pkl'.format(opt.model_name)) # for generating embeddings for all 33 cancers patients
    # data_cv_path_template = os.path.join(opt.dataroot, opt.disease, 'test', '{}_test_10cv_fold_{}.pkl'.format(opt.model_name, '{}')) # current format for classification task
    # data_cv_path_template = os.path.join(opt.dataroot, opt.disease, 'test', '{}_test_combined.pkl'.format(opt.model_name))
elif opt.cancer == 'CPTAC-LUSC':
    print("############# Loading CPTAC LSCC pkl file #############")
    data_cv_path_template = os.path.join(opt.dataroot, opt.disease, opt.cancer, 'test', 'combined_test_combined_padded_norm.pkl')
elif opt.cancer == 'Moffitt-LSCC':
    print("############# Loading Moffitt LSCC pkl file #############")
    data_cv_path_template = os.path.join(opt.dataroot, opt.disease, opt.cancer, 'test', 'allinone_test_combined_padded.pkl')    
else:
    print("############# Loading pkl file for {} #############", opt.cancer)
    data_cv_path_template = os.path.join(opt.dataroot, opt.disease, opt.cancer,'test', '{}_test_combined.pkl'.format(opt.model_name))

for fold in range(1, 11): # 7 folds for CPTAC-LUSC, # 1 folds for TCGA-PRAD, TCGA-TGCT because of imbalanced censor data samples
    data_cv_path = data_cv_path_template
    # uncomment the following if data is in different folds
    # data_cv_path = data_cv_path_template.format(fold)
    print("Loading data from %s" % data_cv_path)
    
    # Use joblib to load the data
    data_cv = load(data_cv_path)    
    data_cv_splits = data_cv['cv_splits']

    ### Main Loop
    for k, data in data_cv_splits.items():
        print("*******************************************")
        print("************** SPLIT (%d) **************" % (k))
        print("*******************************************")

        #### 3.1 Loading Checkpoints
        if opt.cancer is None:
            # load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, '%s_%s_%d_for_test.pt' % (opt.disease, opt.model_name, k))
            # load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, k))
            load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold))
        elif opt.cancer == 'CPTAC-LUSC':
            print("############### Cancer Type is CPTAC-LUSC ####################")
            # load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold)) # for inference on the trained model checkpoint
            load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, opt.cancer, 'Finetuned_Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold)) # for inference on the finetuned model checkpoint
        elif opt.cancer == 'Moffitt-LSCC':
            print("############### Cancer Type is Moffitt-LSCC ####################")
            # load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold)) # for inference on the trained model checkpoint
            load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, opt.cancer, 'Finetuned_Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold)) # for inference on the finetuned model checkpoint
        else:
            # load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold))
            load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, opt.cancer, 'Finetuned_Checkpoint_%s_%s_fold%d_for_test.pt' % (opt.disease, opt.model_name, fold))

        model_ckpt = torch.load(load_path, map_location=device)
        model_state_dict = model_ckpt['model_state_dict']
        if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata
        model = definenet(opt)
        
        if fold==1:
            print(model)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model_state_dict = {fold.replace('module.',''): v for fold, v in model_state_dict.items()}

        print('Loading the model from %s' % load_path)
        model.load_state_dict(model_state_dict)
        loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, pat_features, class_acc_test = test(opt, model, data, 'test', device)
        
        if opt.task == 'surv':
            print("[Final] Test Loss: %.10f" % (loss_test))
            print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            print("[Final] Apply model to testing set: Loss: %.10f" % (loss_test))
            results_test.append(cindex_test)
            features.append(pat_features)

        elif opt.task == 'class':
            print("[Final] Test Loss: %.10f" % (loss_test))
            print("[Final] Apply model to testing set: Classification Accuracy: %.10f" % (class_acc_test))
            logging.info("[Final] Apply model to testing set: Classification Accuracy: %.10f" % (class_acc_test))
            results_test.append(class_acc_test)

        print()
        print('Dumping pkl files of predictions')
        if opt.cancer is None:
            # pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Inference_predictions_%s_%s_fold%d.pkl' % (opt.disease, opt.model_name, k)), 'wb')) # fold-wise
            # pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Inference_predictions_%s_%s.pkl' % (opt.disease, opt.model_name)), 'wb'))
            pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Inference_predictions_%s_%s_33cancers.pkl' % (opt.disease, opt.model_name)), 'wb'))
        else:
            # pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, opt.cancer, 'Inference_predictions_%s_fold%d.pkl' % (opt.cancer, fold)), 'wb'))
            pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, opt.cancer, 'Inference_predictions_%s_fold%d.pkl' % (opt.cancer, fold)), 'wb'))
            
print('Test Split Results:', results_test)
print("Average Test:", np.array(results_test).mean())

print('Dumping the CIndex results in pkl files')
if opt.task == 'class':
    pickle.dump(results_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Inference_Accuracy_%s_%s.pkl' % (opt.disease, opt.model_name)), 'wb'))
else:
    if opt.cancer is None:
        pickle.dump(results_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'Inference_CIndex_%s_%s.pkl' % (opt.disease, opt.model_name)), 'wb'))
    else:
        pickle.dump(results_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, opt.cancer, 'Inference_CIndex_%s.pkl' % (opt.cancer)), 'wb'))

### Save Features
print("Saving Features or Embeddings")
pickle.dump(features, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, opt.disease, 'FINAL_%s_%s_embdgs.pkl' % (opt.disease, opt.model_name)), 'wb'))
print('Done!')