
import numpy as np
import csv
import pickle

def read_data(address):
    IDs = []
    labels = []
    events = []
    lstm = []
    baseline = 0
    with open(address) as csv_file:
    # with open("stage_2_train_labels.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                IDs.append(row[0])
                if row[3] == '' and row[4] == '':
                  labels.append(baseline)
                  baseline += 0.01
                  events.append(0)
                  lstm.append([0, 0, 0, 0])

                else:
                  labels.append(float(row[3]) * float(row[4]))
                  lstm.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
                  events.append(1)

    labels_np = np.array(labels)
    inds = np.argsort(-labels_np)
    np.random.shuffle(inds)
    IDs_sorted = np.array(IDs)[inds]
    events_sorted = np.array(events)[inds]
    lstm_sorted = np.array(lstm)[inds]
    data = {
    'T': inds,
    'X_ID': IDs_sorted,
    'E': events_sorted,
    'X_LSTM': lstm_sorted}
    print('Data Loading complete!')
    return data

def read_pickle(address):
    with open(address, 'rb') as handle:
        xray_dict = pickle.load(handle)
    IDs, events, T, lstm = [], [], [], []
    for p, info_dict in xray_dict.items():
        IDs.append(info_dict['Xray Names'])
        events.append(info_dict['E'])
        T.append(info_dict['T'])
        lstm.append([0,0,0,0])
    inds = np.argsort(-np.array(T))
    IDs_sorted = np.array(IDs)[inds]
    events_sorted = np.array(events)[inds]
    lstm_sorted = np.array(lstm)[inds]
    inds = np.array(T)[inds]
    data = {
    'T': inds,
    'X_ID': IDs_sorted,
    'E': events_sorted,
    'X_LSTM': lstm_sorted}
    print('Data Loading complete!')
    return data

#p_size: number of fix batches from samples
#k_size: number of images drawn from each patients
def prepare_fix_batch(p_size, k_size, address, batch_size):
    data = read_pickle(address)
    X, E, T = data['X_ID'], data['E'], data['T']
    E_pos = np.where(E == 1)
    E_neg = np.where(E == 0)
    E_pos_num = int(np.size(E_pos)/np.size(E) * batch_size)
    E_neg_num = batch_size - E_pos_num
    batches = [stratified_filter(data, E_pos, E_neg, E_pos_num, E_neg_num) for i in range(p_size)]
    print('data generated successfully.')
    # generate indices for fix images set
    for batch in batches:
        inds = []
        for x_list in batch['X']:
            if len(x_list) > k_size:
                inds.append(np.random.choice(range(len(x_list)), size=k_size, replace=False))
            else:
                inds.append(None)
        batch['inds'] = inds
    print('inds generated successfully.')
    return batches


def random_filter(data):
    X, E, T = data['X_ID'], data['E'], data['T']
    batch_indices = np.random.choice(range(len(data['X_ID'])), size=args['batch_size'], replace=False)
    batch_indices.sort()
    X, E, T = X[batch_indices], E[batch_indices], T[batch_indices]
    data_filtered = {'X': X,
                    'E': E,
                    'T': T}
    return data_filtered


def stratified_filter(data, E_pos, E_neg, E_pos_num, E_neg_num):
    X, E, T, L = data['X_ID'], data['E'], data['T'], data['X_LSTM']
    pos_indices = np.random.choice(range(np.size(E_pos)), size=E_pos_num, replace=False)
    neg_indices = np.random.choice(range(np.size(E_neg)), size=E_neg_num, replace=False)
    batch_indices = np.concatenate([E_pos[0][pos_indices], E_neg[0][neg_indices]])
    batch_indices.sort()
    X, E, T, L = X[batch_indices], E[batch_indices], T[batch_indices], L[batch_indices]
    data_filtered = {'X': X,
                    'E': E,
                    'T': T,
                    'L': L}
    return data_filtered

