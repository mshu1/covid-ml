import pickle
import numpy as np
import pandas as pd
import datetime

print('version: 2020 12 18')
with open('Xray_data.pickle', 'rb') as handle:
    xray_dict = pickle.load(handle)
with open('Patient_data.pickle', 'rb') as handle:
    patient_dict = pickle.load(handle)
with open('var_fix_dict.pickle', 'rb') as handle:
    var_fix_dict = pickle.load(handle)

desired = 'ICU Discharge Date'
former = 'ICU Admit Date'
admitted_dict = {}
non_dict = {}
censored_neg = 0
uncensored_neg = 0

##############TIME FIX#############################3

for key, value in patient_dict.items():
  xray_cand = value['Xrays']

  if not (isinstance(value[desired], float) or isinstance(value[desired], str)):
    value[desired] = value[desired] + datetime.timedelta(days=5)
  if not (isinstance(value[former], float) or isinstance(value[former], str)):
    value[former] = value[former] + datetime.timedelta(days=5)

  if isinstance(value[desired], str):
    censored_neg += 1
    continue 
  if pd.isnull(value[desired]):
      if pd.isnull(value[former]) or isinstance(value[former], str):
        censored_neg += 1
        continue
      else:
        flag = 0
        for xray_id in xray_cand:
            x_date = datetime.datetime.strptime(xray_dict[xray_id]['Xray Date'], "%m/%d/%y %H:%M:%S")
            if x_date >= (value[former] - datetime.timedelta(days=7)):
                flag = 1
                if key not in non_dict:
                    non_dict[key] = [xray_id]
                else:
                    non_dict[key].append(xray_id)
        if flag == 0:
            censored_neg += 1
        continue
  if isinstance(value[former], float) or isinstance(value[former], str):
    censored_neg += 1
    continue
  flag = 0
  for xray_id in xray_cand:
      x_date = datetime.datetime.strptime(xray_dict[xray_id]['Xray Date'], "%m/%d/%y %H:%M:%S")
      if x_date < value[desired] and x_date >= (value[former] - datetime.timedelta(days=7)):
        flag = 1
        if key not in admitted_dict:
            admitted_dict[key] = [xray_id]
        else:
            admitted_dict[key].append(xray_id)
  if flag == 0:
    uncensored_neg += 1


print(desired, ':', len(admitted_dict))
print('Censored', ':', len(non_dict))


def generate_dict(uncensor, target_dict):
    date_dict = {}
    censored_neg_add = 0
    for patient, xray_list in target_dict.items():
        date_sort = []
        for xray_id in xray_list:
            start_date = xray_dict[xray_id]['Xray Date'].split(' ')[0]
            end_date = datetime.datetime.strptime(start_date, "%m/%d/%y")
            date_sort.append(end_date)
        date_sort.sort()
        end_time = None
        if uncensor:
            lap = patient_dict[patient][desired] - patient_dict[patient][former]
            end_time = patient_dict[patient][desired]
        else: 
            lap =  date_sort[-1] - patient_dict[patient][former]
            if lap <= datetime.timedelta(days = 0, minutes = 0, seconds = 0):
                censored_neg_add += 1
                print('skipped')
                continue
            end_time = date_sort[-1]
        # print(date_sort[-1], date_sort[0])
        wc_sort = []
        for end_date in date_sort:
            wc = patient + '-20' + end_date.strftime("%y%m%d")
            if wc not in wc_sort:
                wc_sort.append(wc)
        date_dict[patient] = {'Xray Names': wc_sort,
                            'T': int(str(lap).split(" day")[0]),
                            'E': uncensor,
                            'F': var_fix_dict[patient],
                            'Baseline time': patient_dict[patient][former],
                            'End time': end_time}
    return date_dict, censored_neg_add


time_uncensored_dict,  _ = generate_dict(1, admitted_dict)
time_censored_dict, censored_neg_add = generate_dict(0, non_dict)
censored_neg += censored_neg_add
print('After selection: {} uncensored, {} censored'.format(len(time_uncensored_dict), len(time_censored_dict)))
time_censored_dict.update(time_uncensored_dict)
print('Neglected: {} uncensored, {} censored'.format(uncensored_neg, censored_neg))

df = pd.read_csv('all_labs_ct_newid.csv')

with open('ref_dict.pickle', 'rb') as handle:
    ref_dict = pickle.load(handle)

additional_terms = ['vs_hr_hr',
                    'xp_resp_spo2',
                    'xp_resp_rate_pt',
                    'vs_bp_noninvasive (s)',
                    'vs_bp_noninvasive (d)',
                    'vs_bp_noninvasive (m)',
                    'HCO3 (Arterial) - EPOC',
                    'pO2 (Arterial) - EPOC',
                    'pCO2 (Arterial) - EPOC',
                    'pH (Arterial) - EPOC']
ref_base_count = 19
for i, term in enumerate(additional_terms):
    ref_dict[term] = i + ref_base_count
total_count = ref_base_count + len(additional_terms)
print('adding {} terms to reference dictionary...total of {} terms'.format(len(additional_terms), total_count) )

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

for patient_id, patient_info in time_censored_dict.items():
    baseline_time = patient_info['Baseline time'] - datetime.timedelta(days=5) - datetime.timedelta(days=7) 
    end_time = patient_info['End time'] - datetime.timedelta(days=5)
    s = baseline_time.strftime('20'+"%y-%m-%d %H:%M:%S")
    e = end_time.strftime('20'+"%y-%m-%d %H:%M:%S")
    pid = int(patient_id.split('-')[-1])
    lab_data = df.loc[(df['new_patient_id'] == pid) & (df['observation_time'] >= s) & (df['observation_time'] <= e)]
    t = patient_info['T']
    # 19 different observations
    data_array = np.zeros((total_count,2,t+1+7))
    for index, row in lab_data.iterrows():
        if row['observation_type'] in ref_dict:
            if pd.isnull(row['observation_value']):
                print('skipped: ', row['observation_type'], row['observation_value'])
                continue
            if '<' in row['observation_value'] or '>' in row['observation_value']:
                row['observation_value'] = row['observation_value'][1:]
            if not RepresentsFloat(row['observation_value']):
                print('skipped: ', row['observation_type'], row['observation_value'])
                continue
            lab_time = datetime.datetime.strptime(row['observation_time'].split()[0][2:], "%y-%m-%d") #2020-04-24 14:43:00.000
            dif = lab_time - baseline_time
            if dif == datetime.timedelta(days = 0, minutes = 0, seconds = 0):
                data_array[ref_dict[row['observation_type']], 0,0] = row['observation_value']
                data_array[ref_dict[row['observation_type']], 1,0] = 1
            else:
                # print(dif, int(str(dif).split(" day")[0]), t, ref_dict[row['observation_type']])
                day = int(str(dif).split(" day")[0])
                data_array[ref_dict[row['observation_type']], 0,day] = row['observation_value']
                data_array[ref_dict[row['observation_type']], 1,day] = 1
        # else:
        #     print('skpped key:', row['observation_type'])
    print("completed: ", patient_id)
    time_censored_dict[patient_id]['L'] = data_array

AD = time_censored_dict
whole_array = np.zeros((total_count,2,1))
for key, values in AD.items():
    whole_array = np.concatenate([whole_array, values['L']], axis=2)
m_std = np.zeros((whole_array.shape[0], 4))
for i in range(whole_array.shape[0]):
    sub_param = whole_array[i,0][whole_array[i,1]==1]
    if len(sub_param) == 0:
        continue
    m_std[i,0] = sub_param.min()
    m_std[i,1] = sub_param.max()
    m_std[i,2] = sub_param.mean()
    m_std[i,3] = sub_param.std()

for key, values in AD.items():
    values['L'][:, 0, :] = (values['L'][:, 0, :] - m_std[:,0].reshape(-1,1))/(m_std[:,1].reshape(-1,1) - m_std[:,0].reshape(-1,1) + 1e-6)
    values['L'][:, 0, :] = values['L'][:, 0, :] * values['L'][:, 1, :]
    cat = np.concatenate([values['L'][:, 0, :], values['L'][:, 1, :]])
    AD[key]['L'] = cat
AD['min_max_mean_std'] = m_std

print('Normalization Converstion Completed.')

with open('v2/ICU_Discharge_Date_Predict_NEW.pickle', 'wb') as handle:
    pickle.dump(AD, handle, protocol=pickle.HIGHEST_PROTOCOL) 
