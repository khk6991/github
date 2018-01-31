import json
import pandas as pd
import numpy as np
import pickle as pk

path = 'D:\data_set\loldata/'
json_data1 = open(path+'train_data.json').read()
json_data2 = open(path+'test.input.json').read()
json_data3 = open(path+'grading.input.json').read()

train_data = json.loads(json_data1)
test_input = json.loads(json_data2)
test_output = open(path+'test.output').read()
grading_input = json.loads(json_data3)

'''
# 첫 번째 경기 천번째 참가자 & 영웅의 전적
users = []
train_output = []
for gameId, game in enumerate(train_data):
    d = pd.DataFrame([game['teams'][0]['winner'], game['teams'][1]['winner']], index=['100', '200']).T
    train_output.append(np.array(d).tolist()[0])

    for participantsId, participant in enumerate(game['participants']):
        a = pd.DataFrame([participant['summonerId'], participant['championId']], index=['summonerId', 'championId']).T
        b = pd.DataFrame(participant['stats']).drop(['items'], 1).drop_duplicates()
        c = pd.DataFrame(participant['stats']['items'], index=['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6']).T
        abc = pd.concat((a, b, c), 1)
        users.append(np.array(abc).tolist()[0])

users = pd.DataFrame(users, columns=list(abc.keys()))
train_output = pd.DataFrame(train_output, columns=['100', '200'])

users.to_csv(path+'users.csv')
train_output.to_csv(path+'train.output.csv')

item = pd.concat((users['item0'], users['item1'], users['item2'], users['item3'], users['item4'], users['item5'], users['item6'])).drop_duplicates()
item = item.sort_values().reset_index()
del item['index']
item = item.iloc[1:, :]

dummy_item = pd.get_dummies(item[0])

item_dict = {}
for itemidx, item in enumerate(list(dummy_item.keys())):
    item_dict.update({item: np.array(dummy_item)[itemidx].tolist()})

item0 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item0']])
item1 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item1']])
item2 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item2']])
item3 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item3']])
item4 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item4']])
item5 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item5']])
item6 = np.array([item_dict.get(item, list(np.zeros([1, len(item_dict)]).reshape(-1))) for item in users['item6']])

item = item0+item1+item2+item3+item4+item5+item6

users2 = pd.concat((users.iloc[:, :-7], pd.DataFrame(item)), 1)
users2 = users2.sort_values(by=['summonerId', 'championId'])
users2 = users2.reset_index()
del users2['index']

Id = pd.concat((users2['summonerId'], users2['championId']), 1)
Id = np.array(Id.drop_duplicates())

summer_champ = []
for idx, i in enumerate(Id):
    sid = users2[users2['summonerId'] == i[0]]
    fsum = np.mean(np.array(sid[sid['championId'] == i[1]].iloc[:, 2:-212]), 0).tolist()
    dsum = sum(np.array(sid[sid['championId'] == i[1]].iloc[:, -212:])).tolist()
    new_dsum = []
    for j in dsum:
        if j > 0:
            new_dsum.append(1)
        else:
            new_dsum.append(0)
    summer_champ.append([i[0], i[1]]+fsum+new_dsum)
    if idx % 100 == 0:
        print(idx, ',', 'summonerId:', i[0], ',', 'championId:', i[1])
    print(len(Id) == len(summer_champ))

summer_champ_data = pd.DataFrame(summer_champ, columns=list(users2.keys()))
summer_champ_data.to_csv(path+'sum_cham_data.csv', index=None)

summon_champ_dict = {}
for idx, i in enumerate(summer_champ):
    summon_champ_dict.update({str(i[0])+','+str(i[1]): i[2:]})
    if idx % 100 == 0:
        print(str(i[0])+','+str(i[1]))

with open(path+'summon_champ_dict.pickle', 'wb') as f:
    pk.dump(summon_champ_dict, f)
'''
with open(path+'summon_champ_dict.pickle', 'rb') as f:
    summon_champ_dict = pk.load(f)

train_output = pd.read_csv(path+'train.output.csv')

train_input = []
game_time = []
for gameId, game in enumerate(train_data):
    match = []
    for user in game['participants']:
        match.append([str(user['summonerId'])+','+str(user['championId']),  user['teamId']])
    train_input.append([gameId, match])
    game_time.append([gameId, game['matchDuration']])
    if gameId % 100 == 0:
        print(gameId)

test_input2 = []
for gameId, game in enumerate(test_input):
    match = []
    for user in game['participants']:
        match.append([str(user['summonerId'])+','+str(user['championId']), user['teamId']])
    test_input2.append([gameId, match])

train_x = []
for idx, i in enumerate(train_input):
    tmp100 = [summon_champ_dict.get(j[0], np.zeros(223).tolist()) + [0] for j in i[1] if j[1] == 100]
    tmp200 = [summon_champ_dict.get(j[0], np.zeros(223).tolist()) + [1] for j in i[1] if j[1] == 200]
    train_x.append(np.array(tmp100+tmp200).reshape(1, 2240).tolist()[0])
    if idx % 100 == 0:
        print(idx)

train_x = np.array(train_x)
train_y = np.array(train_output.iloc[:, 2:], dtype=int)

test_x = []
for idx, i in enumerate(test_input2):
    tmp100 = [summon_champ_dict.get(j[0], np.zeros(223).tolist()) + [0] for j in i[1] if j[1] == 100]
    tmp200 = [summon_champ_dict.get(j[0], np.zeros(223).tolist()) + [1] for j in i[1] if j[1] == 200]
    test_x.append(np.array(tmp100+tmp200).reshape(1, 2240).tolist()[0])
    if idx % 100 == 0:
        print(idx)

test_x = np.array(test_x)
test_y = np.array(pd.get_dummies(pd.DataFrame(test_output.split())).iloc[:, 1:])


from sklearn import preprocessing

#minmax = preprocessing.MinMaxScaler()
#minmax.fit(train_x)
#train_x = minmax.transform(train_x)
#test_x = minmax.transform(test_x)

stand = preprocessing.StandardScaler()
stand.fit(train_x)
train_x = stand.transform(train_x)
test_x = stand.transform(test_x)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

logit = LogisticRegression()
logit.fit(train_x, train_y)
print(logit.score(test_x, test_y))

rf = RandomForestClassifier(criterion='entropy', n_estimators=100)
rf.fit(train_x, train_y)
print(rf.score(test_x, test_y))

sv = svm.SVC()
sv.fit(train_x, train_y)
print(sv.score(test_x, test_y))

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
print(classification_report(test_y, rf.predict(test_x)))
roc_auc_score(test_y, rf.predict_proba(test_x)[:, 1:])
roc_curve(test_y, rf.predict_proba(test_x)[:, 1:])
plt.plot(roc_curve(test_y, rf.predict_proba(test_x)[:, 1:])[0], roc_curve(test_y, rf.predict_proba(test_x)[:, 1:])[1])

import xgboost
xr = xgboost.XGBClassifier()
xr.fit(train_x, train_y)
xr.score(test_x, test_y)
