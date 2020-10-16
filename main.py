import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import random

random.seed ( 10 )

epoch = 200                     #最多跑的回合數
learning_rate = 0.005           #學習率

#path1 = "example_train.txt"
#path2 = "example_test.txt"
path1 = "train.txt"
path2 = "test.txt"

#處理檔案
data = genfromtxt(path1, delimiter=',', names=('x1', 'x2', 'y'))
test_data = genfromtxt(path2, delimiter=',', names=('x1', 'x2'))


def pla(dataset):
    w = np.array([random.random(), random.random(), random.random()])   #初始化權重
    #print(w)
    for ep in range(epoch):
        flag = True                                                     #測試是否全測資成功
        for cur in data:
            tmp = np.array( [ (1, cur['x1'], cur['x2']), cur['y'] ] )   #將需要內積的項目取出
            if np.sign( np.dot(w, tmp[0]) ) != cur['y']:                #判斷是否符合
                w = w + np.multiply(tmp[0], tmp[1]) * learning_rate     #計算
                flag = False
        if flag == True :
            #print("end in ep:" + str(ep) )
            break
    #print(w)
    return w
    
def print_graphic(w, test_ans):
    plt.xlim(-27,27)                #畫布大小
    plt.ylim(-27,27)
    plt.axhline(0, color= 'gray')   #橫軸
    plt.axvline(0, color= 'gray')   #直軸

    plt.xlabel("x1")
    plt.ylabel("x2")

    x1 = np.linspace(-100,100,100)
    x2 = -w[1]*x1 /w[2]             
    plt.plot(x1,x2)                 #畫出學習完的分隔線

    for i in data:                  #畫出訓練data的分布
        if i['y'] == -1:
            plt.plot(i['x1'],i['x2'],"x", color='r', markersize=4)
        else:
            plt.plot(i['x1'],i['x2'],"o", color='black', markersize=4)
    for i in test_ans[0:]:          #畫出測試data的分布
        if i[1] == -1:
            plt.plot(i[0][1],i[0][2],"^", color='r')
        else: 
            plt.plot(i[0][1],i[0][2],"^", color='black' )

    plt.show()                      

def test(w , test_data):
    list = []
    for cur in test_data:
        tmp = np.array( [ 1, cur['x1'], cur['x2'] ])
        tmp_ans = np.array( [ (1, cur['x1'], cur['x2']), np.sign( np.dot(w, tmp)) ] )   #內積出y並加入回list
        list.append(tmp_ans)
    return list


weight = pla(data)
test_ans = test(weight, test_data)
print("w0 = " + str(weight[0]))
print("w1 = " + str(weight[1]))
print("w2 = " + str(weight[2]))
print("test data result: ")
for obj in test_ans:
    print(str(obj[0]) + " class : " + str(obj[1]))
print_graphic(weight, test_ans)