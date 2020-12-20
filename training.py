import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
import os
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
input_videos_folder='videos/'#path of input video
h5file_name='files/training_datasets.h5' #path of .h5 file
jsonfile_name='files/split' #path of .json file
class Model_Resnet(nn.Module):
    def __init__(self):
        #Building a sequential model based on pretrained Resnet152 .Then excluding last 2 modules to extract features.
        super(Model_Resnet, self).__init__()
        self.fea_type = 'resnet152'
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        resnet = models.resnet152(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]
    def forward(self, x):
        #preproocessing input image and passes through the model to extract features
        x = self.transform(x)
        x = x.unsqueeze(0) 
        x = Variable(x).cuda()
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)
        return res_pool5

def calc_scatters(K):
    #calculating scatter matrix to find covariance
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1); 
    scatters = np.zeros((n, n))
    diagK2 = np.diag(K2)
    i = np.arange(n).reshape((-1,1))
    j = np.arange(n).reshape((1,-1))
    scatters = (K1[1:].reshape((1,-1))-K1[:-1].reshape((-1,1))
                - (diagK2[1:].reshape((1,-1)) + diagK2[:-1].reshape((-1,1)) - K2[1:,:-1].T - K2[:-1,1:]) / ((j-i+1).astype(float) + (j==i-1).astype(float)))
    scatters[j<i]=0
    return scatters

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,out_scatters=None):
    #finding change points
    m = int(ncp) 
    (n, n1) = K.shape
    J = calc_scatters(K)
    if out_scatters != None:
        out_scatters[0] = J
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]
    if backtrack:
        p = np.zeros((m+1, n+1), dtype=int)
    for k in range(1,m+1):
        for l in range((k+1)*lmin, n+1):
            tmin = max(k*lmin, l-lmax)
            tmax = l-lmin+1
            c = J[tmin:tmax,l-1].reshape(-1) + I[k-1, tmin:tmax].reshape(-1)
            I[k,l] = np.min(c)
            if backtrack:
                p[k,l] = np.argmin(c)+tmin
    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]
    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores
def cpd_auto(K, ncp, vmax, desc_rate=1 ):
    #finding change points based on Kernal temporal segmentation method
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False)   
    N = K.shape[0]
    penalties = np.zeros(m+1)
    ncp = np.arange(1, m+1)
    penalties[1:] = (ncp/(2.0*N))*(np.log(float(N)/ncp)+1)
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best)
    return (cps, costs)
  
class Encoder:
    #encoder
    def __init__(self, video_path, save_path): 
        self.resnet = Model_Resnet()
        self.dataset = {}
        self.video_name=video_path.split('/')[-1]   
        self.video_path = video_path
        self.h5_file = h5py.File(save_path, 'w')

        self.video_list = []
        self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        #creating groups for each video
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))

    def _extract_feature(self, frame):
        #extracting features
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        res_pool5 = self.resnet(frame)
        frame_features = res_pool5.cpu().data.numpy().flatten()
        return frame_features

    def _get_change_points(self, video_feat, n_frame, fps):
        # extracting change points and number of frames per segment
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0],change_points,[n_frame-1]))
        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]

            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))
        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0]
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))
        return change_points, n_frame_per_seg
    def _save_dataset(self):
        pass

    def encoder(self):
        #adding contents to the .h5 file
        for video_idx, video_filename in enumerate(self.video_list):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)
            video_basename = os.path.basename(video_path).split('.')[0]
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_list = []
            picks = []
            print(video_filename)
            video_feat = None
            video_feat_for_train = None
            c=0
            user_summary=[]
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                if success:
                    frame_feat = self._extract_feature(frame)
                    print(frame_feat)
                    if frame_idx % 1 == 0:
                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))
                    if video_feat is None:
                        video_feat = frame_feat
                    else:
                        video_feat = np.vstack((video_feat, frame_feat))
                else:
                    break
            video_capture.release()
            change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps)
            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)#features of every 5th frame
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks)) # positions of subsampled frames in original video
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames# number of frames of video
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps #frames per second
            self.h5_file['video_{}'.format(video_idx+1)]['video_name'] = self.video_name #name of input video
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points # change points(indices of segments)
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg #number of frames per segment
#generating .h5 file

h5_gen = Encoder(input_videos_folder,h5file_name)
h5_gen.encoder()
h5_gen.h5_file.close()
print("Encoder process complete!")
from __future__ import print_function
import json
import os
import argparse
import h5py
import math
import numpy as np
import sys
def write_json(obj, fpath):
    #writing in json file
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
def splitting_random(keys, num_videos, num_train):
    #splitting dataset
    train_keys, test_keys = [], []
    random_indexes = np.random.choice(range(num_videos), size=num_train, replace=False)

    for key_index, key in enumerate(keys):
        if key_index in random_indexes:
            train_keys.append(key)
        else:
            test_keys.append(key)
    return train_keys, test_keys

def create():
    #creating json file
    dataset = h5py.File(dataset_path, 'r')
    keys = dataset.keys()
    number_of_videos = len(keys)
    number_of_train = int(math.ceil(number_of_videos * train_percent))
    number_of_test = number_of_videos - number_of_train
    splits = []
    for split_idx in range(num_splits):
        train_keys, test_keys = splitting_random(keys, number_of_videos, number_of_train)

        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys,
        })
    save_path = os.path.join( save_name + '.json')
    write_json(splits, save_path)
    dataset.close()
#splitting process
dataset_path=h5file_name
num_splits=5
train_percent=0.8
save_name=jsonfile_name
create()
print("json file created")

from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from torch.nn import functional as F
class DSN(nn.Module):
    #lstm
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)
    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p
def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    #Computing diversity reward and representativeness reward
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1    
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()
    n = _seq.size(0)
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))   
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0] 
    reward_rep = torch.exp(-dist_mat.mean())
    reward = (reward_div + reward_rep) * 0.5
    return reward
def read_json(fpath):
    #reading from .json file
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj
def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    #implementing knapsack
    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    for i in range(1,n_items+1):
        for w in range(0,capacity+1):
            wi = weights[i-1] 
            vi = values[i-1] 
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                table[i,w] = table[i-1,w]
    picks = []
    K = capacity
    for i in range(n_items,0,-1):
        if keep[i,K] == 1:
            picks.append(i)
            K -= weights[i-1]
    picks.sort()
    picks = [x-1 for x in picks] 
    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks
def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    #Generate video summary 
    n_segs = cps.shape[0]    
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))
    limits = int(math.floor(n_frames * proportion))
    picks = knapsack_dp(seg_score, nfps, n_segs, limits)
    summary = np.zeros((1), dtype=np.float32) 
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))
    summary = np.delete(summary, 0) 
    return summary


def evalt(model, dataset, test_keys, use_gpu): 
    #evaluate the model 
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg'
        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...] 
            seq = torch.from_numpy(seq).unsqueeze(0)
            seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()
            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
     
            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)     

def save_checkpoint(state, fpath='checkpoint.pth.tar'):  
    #saving check points 
    torch.save(state, fpath)
seed=1
torch.manual_seed(seed)
split=5
stepsize=30
max_epoch=60
beta=0.01
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def decoder():
    #decoder
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    dataset = h5py.File(h5file_name, 'r')
    num_videos = len(dataset.keys())
    splits = read_json('files/split.json')
    split = splits[0]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    model = DSN(in_dim=2048, hid_dim=256, num_layers=1, cell='lstm')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=1e-05)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.1)
    start_epoch=0
    model = nn.DataParallel(model).cuda()
    start_time = time.time()
    model.train()
    baselines = {key: 0. for key in train_keys} 
    reward_writers = {key: [] for key in train_keys} 
    for epoch in range(start_epoch,max_epoch):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs) 
        for idx in idxs:
            key = train_keys[idx]
            seq = dataset[key]['features'][...] 
            seq = torch.from_numpy(seq).unsqueeze(0) 
            seq = seq.cuda()
            probs = model(seq)
            cost = beta * (probs.mean() - 0.5)**2 
            m = Bernoulli(probs)
            epis_rewards = []
            for _ in range(5):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                reward = compute_reward(seq, actions, True)
                expected_reward = log_probs.mean() * (reward - baselines[key])
                cost -= expected_reward
                epis_rewards.append(reward.item())
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) 
            reward_writers[key].append(np.mean(epis_rewards))
        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
    write_json(reward_writers, osp.join('', 'files/rewards.json'))
    evalt(model, dataset, test_keys, True)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    model_state_dict = model.module.state_dict() 
    model_save_path = osp.join('files/model_epoch' + str(60) + '.pth.tar')
    save_checkpoint(model_state_dict, model_save_path)
    dataset.close()
decoder()    
