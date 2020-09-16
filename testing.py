

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from torch.nn import functional as F
from torchvision import transforms, models
from torch.autograd import Variable
import os
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
from __future__ import print_function
import os.path as osp
import sys
import time
import datetime
from tabulate import tabulate
import glob

input_videos_folder='videos/'#path of input video

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
        os.mkdir('frames')
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
            video_feat = None
            video_feat_for_train = None
            c=0
            count=0
            user_summary=[]
            base_dir = 'frames/video'+f'{video_idx+1}'
            os.mkdir(base_dir)
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                cv2.imwrite( base_dir+'/' + 'frame%d.jpg' % count, frame)     # save frame as JPEG file
                count += 1
                if success:
                    if c==0:
                        key_frame_id=0
                        keyframe=frame
                        c=c+1
                    frame_feat = self._extract_feature(frame)
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
            for i in range(0,len(change_points)):
                if key_frame_id>=change_points[i][0] and key_frame_id<=change_points[i][1]:
                    intervel=change_points[i]
            for i in tqdm(range(n_frames-1)):
                if i>=intervel[0] and i<=intervel[1]:
                    user_summary.append(1)
                else:
                    user_summary.append(0)
            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)#features of every 5th frame
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks)) # positions of subsampled frames in original video
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames# number of frames of video
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps #frames per second
            self.h5_file['video_{}'.format(video_idx+1)]['video_name'] = self.video_name #name of input video
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points # change points(indices of segments)
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg #number of frames per segment
            self.h5_file['video_{}'.format(video_idx+1)]['user_summary'] =np.array(list(user_summary)) # user summary
            self.h5_file['video_{}'.format(video_idx+1)]['gtscore'] = np.array(list(user_summary)) #importance scores

h5file_name='test_data_.h5' #path of .h5 file
h5_gen = Encoder(input_videos_folder,h5file_name)
h5_gen.encoder()
h5_gen.h5_file.close()
print("Encoder process complete!")

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

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    #Compare machine summary with user summary
    machine_summary = machine_summary.astype(np.float32)
    user_summary = np.array([user_summary])
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1
    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])
    f_scores = []
    prec_arr = []
    rec_arr = []
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)
    final_f_score = np.mean(f_scores)
    final_prec = np.mean(prec_arr)
    final_rec = np.mean(rec_arr)   
    return final_f_score, final_prec, final_rec

def evaluate(model, dataset, test_keys):
    # generating result
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' 
        table = [["No.", "Video", "F-score"]]
        h5_res = h5py.File( resultfile_name, 'w')
        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()
            cps = dataset[key]['change_points'][...]
            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]
            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)
            table.append([key_idx+1, key, "{:.1%}".format(fm)]) 
            h5_res.create_group('video_{}'.format(key_idx+1))   
            h5_res['video_{}'.format(key_idx+1)]['score'] = probs  
            h5_res['video_{}'.format(key_idx+1)]['machine_summary'] = machine_summary 
            h5_res['video_{}'.format(key_idx+1)]['gtscore'] = dataset[key]['gtscore'][...]
            h5_res['video_{}'.format(key_idx+1)]['fm'] = fm 
            ldseg=np.array(os.listdir('frames/video'+f'{key_idx+1}'))
            cnt=0
            count=0
            for i in machine_summary:
                if i==1:
                    os.rename('frames/video'+f'{key_idx+1}'+'/frame'+f'{cnt}'+'.jpg','frames/video'+f'{key_idx+1}'+'/sum_frame'+f'{count}'+'.jpg')
                    count=count+1
                cnt=cnt+1
    print(tabulate(table))
    
    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))
    return mean_fm

resultfile_name='test_result.h5'#result file name
torch.manual_seed(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
torch.cuda.manual_seed_all(1)
dataset = h5py.File(h5file_name, 'r')
keys = dataset.keys()
test_keys=[]
for i in keys:
    test_keys.append(i)
model = DSN(in_dim=2048, hid_dim=256, num_layers=1, cell='lstm')
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=1e-05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
checkpoint = torch.load('model_epoch60.pth.tar')
model.load_state_dict(checkpoint)
evaluate(model, dataset, test_keys)

def frm2video(frm_dir, summary,cc):
    a=0
    k=0
    vid_writer = cv2.VideoWriter(osp.join('summary_videos/',str(cc)+'summary.mp4'),cv2.VideoWriter_fourcc(*'MP4V'),30,(640, 480),)
    for idx, val in enumerate(summary):
        if val == 1:
            try:
                image = cv2.imread(dir+'sum_frame'+str(a)+'.jpg', 1) 
                a=a+1     
                frm = cv2.resize(image, (640, 480))
                vid_writer.write(frm)
            except Exception as e:
                continue
dataset = h5py.File(h5file_name, 'r')
keys = dataset.keys()
test_keys=[]
base_dir ='summary_videos'
os.mkdir(base_dir)
for i in keys:
    test_keys.append(i)
h5_res = h5py.File(resultfile_name, 'r')
cc=1
for i in test_keys:
    summary = h5_res[i]['machine_summary'][...]  
    dir='frames/video'+str(cc)+'/'  
    frm2video(dir, summary,cc)
    cc=cc+1
h5_res.close()
