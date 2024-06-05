import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
import gc

from IPython.display import Image, clear_output
from torch.cuda import memory_allocated, empty_cache
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

import json
from collections import OrderedDict

from datetime import datetime

video_path = 'E:/dataset'

test_video_path = 'test video'
test_video_name = 'bb_1_130317_vehicle_257_27164.mp4'
test_video_file = test_video_path + '/' + test_video_name

model_path = 'model'
model_name = 'LSTM-YOLO.pt'
model_file = model_path + '/' + model_name

log_path = 'logs'
log_file = 'DNN log ' + datetime.today().strftime("%Y-%m-%d %H%M%S") + '.txt'
log_file = os.path.join(log_path, log_file)

"""
정상 데이터셋, 비정상 데이터셋 개수
-1, 0일 경우 해당 폴더(normal, abnormal)에 들어있는 데이터 수 전체
폴더에 들어있는 데이터 개수보다 크게 설정할 경우 자동으로 해당 폴더에 들어있는 데이터 수 전체만큼 설정됨
둘 중 가장 적은 데이터 수만큼 자동으로 랜덤 샘플링하여 데이터셋을 설정
예) 정상 데이터셋 20, 비정상 데이터셋 30으로 설정하면 비정상 데이터도 정상 데이터셋 개수인 20개 만큼만 랜덤 샘플링
"""
# 읽을 정상 데이터셋 개수
normal_len = 10

# 읽을 비정상 데이터셋 개수
abnormal_len = -1

split_ratio = [0.7, 0.15, 0.15] # Training, Validation, Test 데이터셋 비율

BATCH_SIZE = 5
EPOCH = 300
NUM_LAYERS = 8
BBOX_LEN = 6
HIDDEN_SIZE = 8

YOLO_CONFIDENCE = 0.3    # Yolv8 Min Detection confidence check

#YOLO_CLASSES = []
#YOLO_CLASSES = [2, 3, 5, 7] # YOLO CLASSES(0: person, 2: car, 3: motorcycle, 5: bus, 7: truck)
YOLO_CLASSES = [2, 5, 7] # YOLO CLASSES(0: person, 2: car, 5: bus, 7: truck)

YOLO_MODEL = 'yolov8x.pt'
#YOLO_MODEL = 'yolov8s.pt'
#YOLO_MODEL = 'yolov8n.pt'

yolo_model = YOLO(YOLO_MODEL)

PATIENCE = 5
EARLYSTOPPING_ENABLE = True
EARLYSTOPPING_VERBOSE = True

raw_data = []
normal_file_list = []
abnormal_file_list = []

def clear_memory():
    if device != 'cpu':
        empty_cache()
    gc.collect()

def get_bbox_list(video_dir):
    frame_length = 30 # LSTM 모델에 넣을 frame 수
    #print(video_dir)

    bbox_list_list, bbox_list_list_flip = [], []
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(video_dir)
    frame_num = 1

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            """YOLO 바운딩 박스 추출"""
            #results = yolo_model.track(frame, show_labels=False, show_conf=False, verbose=False, persist=True)
            if len(YOLO_CLASSES) > 0:
                if BBOX_LEN == 4:
                    results = yolo_model.track(frame, show_labels=False, show_conf=False, verbose=False, classes=YOLO_CLASSES)
                elif BBOX_LEN >= 5:
                    results = yolo_model.track(frame, show_labels=False, show_conf=False, verbose=False, conf=YOLO_CONFIDENCE, classes=YOLO_CLASSES)
            else:
                if BBOX_LEN == 4:
                    results = yolo_model.track(frame, show_labels=False, show_conf=False, verbose=False)
                elif BBOX_LEN >= 5:
                    results = yolo_model.track(frame, show_labels=False, show_conf=False, verbose=False, conf=YOLO_CONFIDENCE)

            #bboxes = torch.tensor(np.array(results[0].boxes.xywh.cpu()))
            if BBOX_LEN == 4:
                bboxes = results[0].boxes.xyxy.cpu()
                #if len(bboxes) == 0:
                    #bboxes = torch.FloatTensor([0, 0, 0, 0])
                bbox_list_list.append(bboxes)
                bbox_list_list_flip.append(bboxes)
            elif BBOX_LEN == 5:
                if results[0].boxes.id != None:
                    bboxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu()

                    track_ids = track_ids.unsqueeze(1)

                    combined_tensor = torch.cat((track_ids, bboxes), dim=1)

                    bbox_list_list.append(combined_tensor)
                    bbox_list_list_flip.append(combined_tensor)
            elif BBOX_LEN == 6:
                if results[0].boxes.id != None:
                    bboxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu()

                    track_ids = track_ids.unsqueeze(1)

                    frame_tensor = [frame_num for _ in range(len(track_ids))]
                    frame_tensor = torch.tensor(frame_tensor)
                    frame_tensor = frame_tensor.unsqueeze(1)

                    combined_tensor = torch.cat((frame_tensor, track_ids), dim=1)
                    combined_tensor = torch.cat((combined_tensor, bboxes), dim=1)

                    bbox_list_list.append(combined_tensor)
                    bbox_list_list_flip.append(combined_tensor)
                    frame_num += 1


        else:
            break

    """부족한 프레임 수 맞추기"""
    """
    if len(bbox_list_list_flip) < 15:
        return False, False
    elif len(bbox_list_list_flip) < frame_length:
        f_ln = frame_length - len(bbox_list_list_flip)
        for _ in range(f_ln):
            bbox_list_list.append(bbox_list_list[-1])
            bbox_list_list_flip.append(bbox_list_list_flip[-1])
    """

    cap.release()
    cv2.destroyAllWindows()

    for idx, raw in enumerate(bbox_list_list):
        if len(raw) == 0:
            bbox_list_list.pop(idx)

    for idx, raw in enumerate(bbox_list_list_flip):
        if len(raw) == 0:
            bbox_list_list_flip.pop(idx)

    return bbox_list_list, bbox_list_list_flip

def padding_raw_data(raw_data):
    # 시퀀스 길이 리스트
    seq_lengths = []
    
    for raw in raw_data:
        f_lengths = [len(raw['value'])]
        bb_lengths = []
        for r in raw['value']:
            bb_lengths.append(len(r))
        f_lengths.append(bb_lengths)
        seq_lengths.append(f_lengths)
        
    #print(seq_lengths)

    max_f_length = max(item[0] for item in seq_lengths)
    max_b_length = max(max(item[1]) for item in seq_lengths)
    
    #print('max_f_length = {}'.format(max_f_length))
    #print('max_b_length = {}'.format(max_b_length))

    padded_f_list = []
    padded_b_len_list = []

    for raw in raw_data:
        padded_b_list = []
        for idx, rb in enumerate(raw['value']):
            padded_b = torch.zeros(max_b_length, BBOX_LEN)
            #if bl != 0:
                #print(len(rb), bl)
                #padded_b[:bl] = rb
            padded_b[:len(rb)] = rb
            padded_b_list.append(padded_b)
        padded_b_list = torch.stack(padded_b_list)
        #print(len(padded_b_list))
        padded_f = torch.zeros(max_f_length, max_b_length, BBOX_LEN)
        #print(len(padded_f))
        #padded_f = []
        padded_f[:len(raw['value'])] = padded_b_list
        padded_f_list.append(padded_f)
    
    for seq_len in seq_lengths:
        padded_b_len = [0] * max_f_length
        padded_b_len[:len(seq_len[1])] = seq_len[1]
        padded_b_len_list.append(padded_b_len)

    padded_f_list = torch.stack(padded_f_list)

    if padded_f_list.dim() == 4:
        padded_f_list = padded_f_list.reshape(len(padded_f_list), len(padded_f_list[0]) * len(padded_f_list[0][0]) * len(padded_f_list[0][0][0]))
        padded_b_len = []
        
        for padded_f in padded_f_list:
            is_zero = (padded_f == 0).all(dim=0)
            #zero_count = is_zero.sum().item()
            non_zero_count = (~is_zero).sum().item()
            padded_b_len.append(non_zero_count)
            #b_sum = sum(b_len)
            #padded_b_len.append(b_sum)
        padded_b_len_list = padded_b_len
    elif padded_f_list.dim() == 3:
        padded_f_list = padded_f_list.reshape(len(padded_f_list), len(padded_f_list[0]) * len(padded_f_list[0][0]))
        padded_b_len = []

        for padded_f in padded_f_list:
            is_zero = (padded_f == 0).all(dim=0)
            #zero_count = is_zero.sum().item()
            non_zero_count = (~is_zero).sum().item()
            padded_b_len.append(non_zero_count)
            #b_sum = sum(b_len)
            #padded_b_len.append(b_sum)
        padded_b_len_list = padded_b_len

    padded_b_len_list = torch.tensor(padded_b_len_list)

    padded_raw = []
    for idx, raw_d in enumerate(padded_f_list):
        #print(raw_d.shape)
        padded_raw.append({'key':raw_data[idx]['key'], 'value':raw_d})
        #padded_raw.append(raw_d)

    return padded_raw, padded_b_len_list, max_f_length, max_b_length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
"""
class BBoxDataset(Dataset):
    def __init__(self, seq_list):
        self.X = []
        self.y = []
        for dic in seq_list:
            #print(dic.__str__())
            self.y.append(dic['key'])
            self.X.append(dic['value'])
    
    def __getitem__(self, index):
        #data = self.X[index][0]
        #len = self.X[index][1]
        data = self.X[index]
        label = self.y[index]
        data = torch.tensor(np.array(data))
        if data.dim() == 3:
            data = data.reshape(len(data) * len(data[index]), BBOX_LEN)
        #print(data.shape)
        #print(label)
        #return torch.tensor(np.array(data)), torch.tensor(np.array(len)), torch.tensor(np.array(int(label)))
        return data, torch.tensor(np.array(int(label)))
    
    def __len__(self):
        return len(self.X)
"""
class BBoxDataset(Dataset):
    def __init__(self, seq_list, padding_func):
        data, self.len, self.f_len, self.b_len = self.pad_data(seq_list, padding_func)
        if BBOX_LEN == 6:
            self.b_len *= self.f_len
        self.X = []
        self.y = []
        for dic in data:
            #print(dic.__str__())
            self.y.append(dic['key'])
            self.X.append(dic['value'])

    def pad_data(self, seq_list, padding_func):
        padded_data, len, f_len, b_len = padding_func(seq_list)

        return padded_data, len, f_len, b_len
    
    def __getitem__(self, index):
        #data = self.X[index][0]
        #len = self.X[index][1]
        data = self.X[index]
        label = self.y[index]
        length = self.len[index]
        #print(self.flen[index])
        #print(length)
        data = torch.tensor(np.array(data))
        #print(data.shape)
        #print(label)
        #return torch.tensor(np.array(data)), torch.tensor(np.array(len)), torch.tensor(np.array(int(label)))
        return data, torch.tensor(np.array(int(label))), length
    
    def __len__(self):
        return len(self.X)
    
class LiquidNewralNetwork(nn.Module):
    def __init__(self, input_len=1):
        super(LiquidNewralNetwork, self).__init__()
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Linear(BBOX_LEN * input_len, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        ))
        for _ in range(NUM_LAYERS - 2):
            self.layers.append(nn.Sequential(
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            ))
        self.layers.append(nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Sigmoid()
        ))
        self.layers = nn.ModuleList(self.layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < NUM_LAYERS - 1:
                x = layer(x)
            else:
                x = layer(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, isenable=True, verbose=False):
        self.patience = patience
        self.isenable = isenable
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, model, val_loss):
        if self.isenable:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_model_wts = model.state_dict()
            elif val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = model.state_dict()
                self.counter = 0
            elif val_loss >= self.best_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    model.load_state_dict(self.best_model_wts)
                    if self.verbose:
                        print(f'Restoring best model weights from epoch with validation loss: {np.round(self.best_loss, 3)}')


raw_data = []

#video_path = 'dataset'
normal_file_list = []
abnormal_file_list = []
for fold in os.listdir(video_path):
    dataset_dir = os.path.join(video_path, fold)
    for file_name in os.listdir(dataset_dir):
        if fold == 'normal':
            normal_file_list.append(file_name)
        else:
            abnormal_file_list.append(file_name)

normal_file_len = len(normal_file_list) if normal_len <= 0 else min(len(normal_file_list), normal_len)
abnormal_file_len = len(abnormal_file_list) if abnormal_len <= 0 else min(len(abnormal_file_list), abnormal_len)

print('normal_file_len = {}, abnormal_file_len = {}'.format(normal_file_len, abnormal_file_len))

if normal_file_len != abnormal_file_len:
    min_len = min(normal_file_len, abnormal_file_len)

    if min_len == len(normal_file_list):
        abnormal_file_list = random.sample(abnormal_file_list, min_len)
    elif min_len == len(abnormal_file_list):
        normal_file_list = random.sample(normal_file_list, min_len)
    else:
        normal_file_list = random.sample(normal_file_list, min_len)
        abnormal_file_list = random.sample(abnormal_file_list, min_len)

print('normal dataset')
for normal_file_name in tqdm(normal_file_list):
#for normal_file_name in normal_file_list:
    label = 0
    video_dir = os.path.join(video_path, 'normal', normal_file_name)
    if os.path.exists(video_dir):
        bbox_data_n, _ = get_bbox_list(video_dir)
        if len(bbox_data_n) > 0:
            raw_data.append({'key':label, 'value':bbox_data_n})
clear_memory()

print('abnormal dataset')
for abnormal_file_name in tqdm(abnormal_file_list):
#for abnormal_file_name in abnormal_file_list:
    label = 1
    video_dir = os.path.join(video_path, 'abnormal', abnormal_file_name)
    if os.path.exists(video_dir):
        bbox_data_n, _ = get_bbox_list(video_dir)
        if len(bbox_data_n) > 0:
            raw_data.append({'key':label, 'value':bbox_data_n})
clear_memory()
"""
for fold in os.listdir(video_path):
    if fold == 'normal':
        label = 0
        print('normal')
    else:
        label = 1
        print('abnormal')
    for video_name in os.listdir(video_path + '/' + fold):
        bbox_data_n, bbox_data_f = get_bbox_list('{}/{}'.format(video_path + '/' + fold, video_name))
        #bbox_data_n, bbox_data_f = padding_list(bbox_data_n, bbox_data_f)
        #seq_list_n = [bbox_data_n, bbox_len_n]
        #seq_list_f = [bbox_data_f, bbox_len_f]
        seq_list_n = bbox_data_n
        seq_list_f = bbox_data_f
        raw_data.append({'key':label, 'value':seq_list_n})
        #raw_data.append({'key':label, 'value':seq_list_f})
"""
random.shuffle(raw_data)
#raw_data, f_legnths, b_lengths = padding_raw_data(raw_data)

train_len = int(len(raw_data) * split_ratio[0])
val_len = int(len(raw_data) * split_ratio[1])
test_len = len(raw_data) - train_len - val_len

print('{}: {}, {}, {}'.format(len(raw_data), train_len, val_len, test_len))

train_dataset = BBoxDataset(raw_data, padding_raw_data)
train_data, valid_data, test_data = random_split(train_dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
test_data_path = 'test data'

# 모델 초기화
def init_model():
    global net, loss_fn, optim
    global early_stopping

    plt.rc('font', size = 10)
    net = LiquidNewralNetwork(train_dataset.b_len).to(device)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optim = Adam(net.parameters(), lr=0.0001)
    early_stopping = EarlyStopping(patience=PATIENCE, isenable=EARLYSTOPPING_ENABLE, verbose=EARLYSTOPPING_VERBOSE)

# epoch 카운터 초기화
def init_epoch():
    global epoch_cnt
    epoch_cnt = 0

# 모든 Log를 초기화
def init_log():
    global log_stack, iter_log, tloss_log, tacc_log, vloss_log, vacc_log, time_log
    plt.rc('font', size = 10)
    iter_log, tloss_log, tacc_log, vloss_log, vacc_log = [], [], [], [], []
    time_log, log_stack = [], []

def record_train_log(_tloss, _tacc, _time):
    # Train Log 기록
    time_log.append(_time)
    tloss_log.append(_tloss)
    tacc_log.append(_tacc)
    iter_log.append(epoch_cnt)

def record_valid_log(_vloss, _vacc):
    # Validation Log 기록
    vloss_log.append(_vloss)
    vacc_log.append(_vacc)

def last(log_list):
    # last 안의 마지막 숫자를 반환(print_log 함수에서 사용)
    if len(log_list) > 0:
        return log_list[len(log_list) - 1]
    else:
        return -1

def print_log(isfinished=False, test_acc=0.0, test_loss=0.0):
    # 학습 추이 출력 : 소숫점 3자리까지
    train_loss = round(float(last(tloss_log)), 3)
    train_acc = round(float(last(tacc_log)), 3)
    val_loss = round(float(last(vloss_log)), 3)
    val_acc = round(float(last(vacc_log)), 3)
    time_spent = round(float(last(time_log)), 3)

    log_str = 'Epoch: {:3} | T_Loss {:5} | T_Acc {:5} | V_Loss {:5} | V_Acc {:5} | {:5}'.format(last(iter_log), train_loss, train_acc, val_loss, val_acc, time_spent)
    print(log_str)

    log_stack.append(log_str)
    
    if isfinished == True:
        # 학습 추이 그래프 출력
        hist_fig, loss_axis = plt.subplots(figsize=(10, 3), dpi=99)
        hist_fig.patch.set_facecolor('white')

        # Loss Line 구성
        loss_t_line = plt.plot(iter_log, tloss_log, label='Train_Loss', color='red', marker='o')
        loss_v_line = plt.plot(iter_log, vloss_log, label='Valid_Loss', color='blue', marker='s')
        loss_axis.set_xlabel('epoch')
        loss_axis.set_ylabel('loss')

        # Acc, Line 구성
        acc_axis = loss_axis.twinx()
        acc_t_line = acc_axis.plot(iter_log, tacc_log, label='Train_Acc', color='red', marker='+')
        acc_v_line = acc_axis.plot(iter_log, vacc_log, label='Valid_Acc', color='blue', marker='x')
        acc_axis.set_xlabel('epoch')
        acc_axis.set_ylabel('accuracy')
        acc_axis.set_ylim(0, 1.0)

        # 그래프 출력
        #hist_lines = acc_t_line + acc_v_line
        hist_lines = acc_t_line + acc_v_line + loss_t_line + loss_v_line
        acc_axis.legend(hist_lines, [l.get_label() for l in hist_lines])
        acc_axis.grid()
        plt.title('Learning history until epoch {}, test_loss = {}, test_acc = {}'.format(last(iter_log), test_loss, test_acc))
        plt.draw()
        plt.show()

        # 텍스트 로그 출력
        """
        clear_output(wait=True)
        for idx in reversed(range(len(log_stack))):
            print(log_stack[idx])
        """
    
# 하이퍼파라미터 등 최종 로그 저장하기
def write_log(log_path, log_file, test_acc, test_loss):
    try:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    except OSError:
        print("ERROR: Failed to create {} directory.".format(log_path))

    with open(log_file, 'w') as fp:
        fp.write('normal_file_len = {}'.format(normal_file_len) + '\n')
        fp.write('abnormal_file_len = {}'.format(abnormal_file_len) + '\n')
        fp.write('split ratio = {}'.format(split_ratio) + '\n')
        fp.write('datasets = {}: {}, {}, {}'.format(len(raw_data), train_len, val_len, test_len) + '\n\n')

        fp.write('BATCH_SIZE = {}'.format(BATCH_SIZE) + '\n')
        fp.write('EPOCH = {}'.format(EPOCH) + '\n')
        fp.write('NUM_LAYERS = {}'.format(NUM_LAYERS) + '\n')
        fp.write('BBOX_LEN = {}'.format(BBOX_LEN) + '\n')
        fp.write('HIDDEN_SIZE = {}'.format(HIDDEN_SIZE) + '\n\n')

        fp.write('PATIENCE = {}'.format(PATIENCE) + '\n')
        fp.write('EARLYSTOPPING_ENABLE = {}'.format(EARLYSTOPPING_ENABLE) + '\n')
        fp.write('EARLYSTOPPING_VERBOSE = {}'.format(EARLYSTOPPING_VERBOSE) + '\n\n')

        fp.write('YOLO_MODEL = {}'.format(YOLO_MODEL) + '\n')
        fp.write('YOLO_CONFIDENCE = {}'.format(YOLO_CONFIDENCE) + '\n')
        fp.write('YOLO_CLASSES = {}'.format(YOLO_CLASSES) + '\n\n')

        for line in log_stack:
            fp.write(line + '\n')
        fp.write('test_acc = {}'.format(test_acc) + '\n')
        fp.write('test_loss = {}'.format(test_loss) + '\n\n')

        fp.write('=========== normal file list ===========\n')
        for line in normal_file_list:
            fp.write(line + '\n')
        fp.write('========================================\n')
        fp.write('\n')

        fp.write('=========== abnormal file list ===========\n')
        for line in abnormal_file_list:
            fp.write(line + '\n')
        fp.write('==========================================\n')

# 학습 알고리즘
def epoch(data_loader, mode = 'train'):
    global epoch_cnt
    
    # 사용되는 변수 초기화
    iter_loss, iter_acc, last_grad_performed = [], [], False
    correct_predictions, total_predictions = 0, 0
    running_loss = 0.0

    # 1 iteration 학습 알고리즘(for문을 나오면 1 epoch 완료)
    for _data, _label, _length in data_loader:
        data, label, length = _data.to(device), _label.type(torch.LongTensor).to(device), _length.type(torch.int64)

        #print('data.shape = {}'.format(data.shape))
        #print('label.shape = {}'.format(label.shape))
        #print('length.shape = {}'.format(length.shape))

        # 1. Feed-forward
        if mode == 'train':
            net.train()
        else:
            # 학습때만 쓰이는 Dropout, Batch Mormalization을 미사용
            net.eval()

        #result = net(data, length) # 1 Batch에 대한 결과가 모든 Class에 대한 확률값으로
        result = net(data) # 1 Batch에 대한 결과가 모든 Class에 대한 확률값으로
        #_, out = torch.max(result, 1) # result에서 최대 확률값을 기준으로 예측 class 도출( _ : 값 부분은 필요 없음, out : index 중 가장 큰 하나의 데이터)
        result = result.squeeze(dim=1)
        label = label.float().view(-1)

        assert torch.all(label >= 0) and torch.all(label <= 1), f"Invalid label: {label}"  # 라벨 값이 0 또는 1인지 확인
        assert torch.all(result >= 0) and torch.all(result <= 1), f"Invalid result: {result}"  # 출력 값이 0 또는 1인지 확인

        #formated_result = [f'{t.item():.4f}' for t in result]
        #print('result = {}, label = {}'.format(formated_result, label))

        # 2. Loss 계산
        loss = loss_fn(result, label) # GT 와 Label 비교하여 Loss 산정
        iter_loss.append(loss.item()) # 학습 추이를 위하여 Loss를 기록

        # 3. 역전파 학습 후 Gradient Descent
        if mode == 'train':
            optim.zero_grad() # 미분을 통해 얻은 기울기를 초기화 for 다음 epoch
            loss.backward() # 역전파 학습
            optim.step() # Gradient Descent 수행
            last_grad_performed = True # for문을 나가면 epoch 카운터 += 1

        # 4. 정확도 계산
        predicted  = (result >= 0.5).float()
        correct_predictions = (predicted == label).sum().item()
        total_predictions = label.size(0)
        acc_partial = correct_predictions / total_predictions
        #print('correct_predictions = {}, total_predictions = {}, acc_partial = {}'.format(correct_predictions, total_predictions, acc_partial))
        iter_acc.append(acc_partial) # 학습 추이를 위하여 Acc. 기록

    # 역전파 학습 후 Epoch 카운터 += 1
    if last_grad_performed:
        epoch_cnt += 1

    clear_memory()

    # loss와 acc의 평균값 for 학습추이 그래프, 모든 GT와 Label 값 for 컨퓨전 매트릭스
    return np.average(iter_loss), np.average(iter_acc)

def epoch_not_finished():
    # 에폭이 끝남을 알림
    return epoch_cnt < maximum_epoch

# Training initialization
init_model()
init_epoch()
init_log()
maximum_epoch = EPOCH

# Training iteration

while epoch_not_finished():
    start_time = time.time()

    tloss, tacc = epoch(train_loader, mode = 'train')

    end_time = time.time()
    time_taken = end_time - start_time
    record_train_log(tloss, tacc, time_taken)

    with torch.no_grad():
        vloss, vacc = epoch(val_loader, mode = 'val')
        record_valid_log(vloss, vacc)

    print_log()
    
    with torch.no_grad():
        early_stopping(net, vloss)
        if early_stopping.early_stop:
            print('Early stopping')
            maximum_epoch = epoch_cnt
    
print('\n Training completed!')

# 정확도 검증
with torch.no_grad():
    test_loss, test_acc = epoch(test_loader, mode = 'test')
    test_acc = round(test_acc, 4)
    test_loss = round(test_loss, 4)
    print('Test Acc.: {}'.format(test_acc))
    print('Test Loss: {}'.format(test_loss))

cv2.destroyAllWindows()
cap = cv2.VideoCapture(test_video_file)
img_list = []

if cap.isOpened():

    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (640, 640))
            img_list.append(img)
            # cv2_imshow(img)
            # cv2.waitKey(1)
        else:
            break

cap.release()
cv2.destroyAllWindows()

print('저장된 frame의 개수: {}'.format(len(img_list)))

"""Yolov5 + Mediapipe Version"""

net.eval()

frame_length = 30 # frame 상태를 표시할 길이
out_img_list = []
dataset = []
status = 'None'
#pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=n_CONFIDENCE)
print('시퀀스 데이터 분석 중...')

cur_frm = 0
xy_list_list = []
for frame_num, img in enumerate(tqdm(img_list)):
    res = yolo_model(img)
    if len(YOLO_CLASSES) > 0:
        if BBOX_LEN == 4:
            results = yolo_model.track(img, show_labels=False, show_conf=False, verbose=False, classes=YOLO_CLASSES)
        elif BBOX_LEN >= 5:
            results = yolo_model.track(img, show_labels=False, show_conf=False, verbose=False, conf=YOLO_CONFIDENCE, classes=YOLO_CLASSES)
    else:
        if BBOX_LEN == 4:
            results = yolo_model.track(img, show_labels=False, show_conf=False, verbose=False)
        elif BBOX_LEN >= 5:
            results = yolo_model.track(img, show_labels=False, show_conf=False, verbose=False, conf=YOLO_CONFIDENCE)

    if BBOX_LEN == 4:
        bboxes = results[0].boxes.xyxy.cpu()
        #if len(bboxes) == 0:
            #bboxes = torch.FloatTensor([0, 0, 0, 0])
        xy_list_list.append(bboxes)

        nms_human = len(bboxes)
        if nms_human > 0:
            for bbox in bboxes:
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)

                #if fram_num >= frame_length:
                if len(xy_list_list) >= frame_length:
                    dataset = []
                    dataset.append({'key': 0, 'value': xy_list_list})
                    dataset = BBoxDataset(dataset, padding_raw_data)
                    b_len = dataset.b_len
                    dataset = DataLoader(dataset)
                    #xy_list_list = []

                    net.layers[0] = nn.Sequential(
                        nn.Linear(BBOX_LEN * b_len, HIDDEN_SIZE),
                        nn.LeakyReLU(),
                        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
                    )
                    net.to(device)

                    for data, label, length in dataset:
                        data, length = data.to(device), length.type(torch.int64)
                        with torch.no_grad():
                            result = net(data, length)
                            #result = net(data)
                            #_, out = torch.max(result, 1)
                            predicted  = (result >= 0.5).float()
                            if predicted.item() == 0: status = 'Normal'
                            else: status = 'Abnormal'
    elif BBOX_LEN == 5:
        if results[0].boxes.id != None:
            bboxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu()

            track_ids = track_ids.unsqueeze(1)

            combined_tensor = torch.cat((track_ids, bboxes), dim=1)

            xy_list_list.append(combined_tensor)

            nms_human = len(bboxes)
            if nms_human > 0:
                for bbox in bboxes:
                    start_point = (int(bbox[0]), int(bbox[1]))
                    end_point = (int(bbox[2]), int(bbox[3]))
                    img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)

                    #if fram_num >= frame_length:
                    if len(xy_list_list) >= frame_length:
                        dataset = []
                        dataset.append({'key': 0, 'value': xy_list_list})
                        dataset = BBoxDataset(dataset, padding_raw_data)
                        b_len = dataset.b_len
                        dataset = DataLoader(dataset)
                        #xy_list_list = []

                        net.layers[0] = nn.Sequential(
                            nn.Linear(BBOX_LEN * b_len, HIDDEN_SIZE),
                            nn.LeakyReLU(),
                            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
                        )
                        net.to(device)

                        for data, label, length in dataset:
                            data, length = data.to(device), length.type(torch.int64)
                            with torch.no_grad():
                                #result = net(data, length)
                                result = net(data)
                                _, out = torch.max(result, 1)
                                if out.item() == 0: status = 'Normal'
                                else: status = 'Abnormal'
    elif BBOX_LEN == 6:
        if results[0].boxes.id != None:
            bboxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu()

            track_ids = track_ids.unsqueeze(1)

            frame_tensor = [frame_num for _ in range(len(track_ids))]
            frame_tensor = torch.tensor(frame_tensor)
            frame_tensor = frame_tensor.unsqueeze(1)

            combined_tensor = torch.cat((frame_tensor, track_ids), dim=1)
            combined_tensor = torch.cat((combined_tensor, bboxes), dim=1)

            xy_list_list.append(combined_tensor)

            nms_human = len(bboxes)
            if nms_human > 0:
                for bbox in bboxes:
                    start_point = (int(bbox[0]), int(bbox[1]))
                    end_point = (int(bbox[2]), int(bbox[3]))
                    img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)

                    #if fram_num >= frame_length:
                    if cur_frm >= frame_length:
                        dataset = []
                        dataset.append({'key': 0, 'value': xy_list_list})
                        dataset = BBoxDataset(dataset, padding_raw_data)
                        b_len = dataset.b_len
                        dataset = DataLoader(dataset)
                        #xy_list_list = []

                        net.layers[0] = nn.Sequential(
                            nn.Linear(BBOX_LEN * b_len, HIDDEN_SIZE),
                            nn.LeakyReLU(),
                            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
                        )
                        net.to(device)

                        for data, label, length in dataset:
                            data, length = data.to(device), length.type(torch.int64)
                            with torch.no_grad():
                                #result = net(data, length)
                                result = net(data)
                                #_, out = torch.max(result, 1)
                                predicted  = (result >= 0.5).float()
                                if predicted.item() == 0: status = 'Normal'
                                else: status = 'Abnormal'
                        cur_frm = 0

    cur_frm += 1

    cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    out_img_list.append(img)

# 테스트 원본 영상 내보내기
filename = './output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 3
frameSize = (640, 640)
isColor = True
out = cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor)
for out_img in out_img_list:
    out.write(out_img)
out.release()

# 모델 저장하기
try:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
except OSError:
    print("ERROR: Failed to create {} directory.".format(model_path))

torch.save(net.state_dict(), model_file)
print('{}, {}, {}'.format(train_len, val_len, test_len))
print_log(isfinished=True, test_acc=test_acc, test_loss=test_loss)

# 하이퍼파라미터 등 최종 로그 저장하기
print("write log...")
write_log(log_path=log_path, log_file=log_file, test_acc=test_acc, test_loss=test_loss)
