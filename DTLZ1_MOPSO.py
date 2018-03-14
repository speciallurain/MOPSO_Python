#--coding:utf-8--
#!/usr/bin/python
# -*- coding: utf-8 -*-
#MOPSO算法 by Lo Rain ,qq:771527850,E-mail:luyueliang423@163.com
import math
import numpy as np
import random
import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
total_archive_Cost=[]
total_archive_Position=[]


def Dominates(x, y):
    # b=all([x[0]<=y[0],x[1]<=y[1]])and any([x[0]<y[0],x[1]<y[1]])
    b = 0
    if x[0] <= y[0]:
        if x[1] <= y[1]:
            if x[2] <= y[2]:
                if x[0] < y[0]:
                    b = 1
                elif x[1] < y[1]:
                    b = 1
                elif x[2] < y[2]:
                    b = 1
    return b
def distance(x, y):
    t = 0
    for i in range(len(x)):
        t = t + (x[i] -y[i]) ** 2
    dis = math.sqrt(t)
    return dis
def Dominates2(x, y):
    # b=all([x[0]<=y[0],x[1]<=y[1]])and any([x[0]<y[0],x[1]<y[1]])
    b = 0
    if x[0] < y[0]:
        b = 1
    if x[1] < y[1]:
        b = 1
    if x[2] < y[2]:
        b = 1
    # 若b=1则，x支配y
    return b
def select_delect(archive_Cost):
    mindistance = 10000
    Number = numpy.zeros([len(archive_Cost), 4])
    a=0
    for i in range(len(archive_Cost)):
        Number[i][0] = archive_Cost[i][0]
        Number[i][1] = archive_Cost[i][1]
        Number[i][2] = archive_Cost[i][2]
        Number[i][3] = a
        a += 1
    SeleteCost = sorted(Number, key=itemgetter(0))
    Dis = []
    Num = []
    for l in range(len(SeleteCost)):
        if l == (len(SeleteCost) - 1):
            break
        ka = distance(SeleteCost[l][:3], SeleteCost[l + 1][:3])
        kn = SeleteCost[l][3]
        Dis.append(ka)
        Num.append(kn)
    for k in range(len(Dis)):
        if k == len(Dis) - 1:
            break
        SumDistance = Dis[k] + Dis[k + 1]
        if SumDistance < mindistance:
            mindistance = SumDistance
            tt = Num[k + 1]
    return int(tt)
def CostFunction(x):  # fitness function for two objects
    n = 0
    for i in range(len(x)):
        if i > 1:
            n = (x[i] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i] - 0.5))
    g = 100 * (len(x) - 3 + n)
    # f1=(0.5*t*(1+g))
    # f2=(0.5*t2*(1+g)*(1-x[len(x)-2]))
    # f3= (0.5*t3*(1+g)*(1-x[len(x)-3]))

    f1 = 0.5 * x[0] * x[1] * (1 + g)
    f2 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
    f3 = 0.5 * (1 - x[0]) * (1 + g)
    # print f2,f3
    return [f1, f2, f3]
def DetermineDomination(pop_Cost):  # for determine the dominatied pop in the orginal pop
    n = len(pop_Cost)
    pop_IsDominated1 = []
    for i in range(n):
        pop_IsDominated1.append(0)
    for i in range(n):
        # if pop_IsDominated[i]:
        #    continue
        for j in range(n):
            if j != i:
                if Dominates(pop_Cost[j], pop_Cost[i]):
                    pop_IsDominated1[i] = 1  # if the pop dominatied we write the tital "1"
                    break
    return pop_IsDominated1
for test2 in range(100):
    print test2
    unfit=0.1#带优化的目标函数
    N=20#内部种群数目
    Nup=200#非劣质解集
    cmax=2.0#学习因子
    cmin=1.0
    w=0.4#惯性权重
    M=50#最大迭代次数
    D=30#问题的维度

    lb=numpy.zeros(D)#每一维的上界
    ub=numpy.ones(D)#每一维的下边界
    pop_Position=numpy.zeros([N,D])
    v=numpy.zeros([N,D])
    for j in range(N):
        for i in range(D):
            pop_Position[j][i]=lb[i]+(ub[i]-lb[i])*random.random()
            v[j][i]=(ub[i]-lb[i])*random.random()*0.5
    pop_Cost = []
    for i in range(N):
        pop_Cost.append(CostFunction(pop_Position[i]))
    vmax=[]
    archive_Position=[]
    archive_Cost=[]
    vmin=[]
    for i in range(D):
        vmax.append((ub[i]-lb[i])*0.5)
        vmin.append(-(ub[i]-lb[i])*0.5)
    ndpop_Position = []  # clear the transfer matrix
    ndpop_Cost = []

    # pop_IsDominated = DetermineDomination(pop_Position)
    # for i in range(len(pop_Position)):
    #     if pop_IsDominated[i] == 0:
    #         ndpop_Position.append(pop_Position[i])
    #         ndpop_Cost.append(pop_Cost[i])
    #
    # for kt in range(len(ndpop_Cost)):
    #     archive_Position.append(ndpop_Position[kt])
    #     archive_Cost.append(ndpop_Cost[kt])

    Pgbest=1.0/len(archive_Cost)
    kl=random.random()
    for i in range(len(archive_Cost)):
        if kl<Pgbest:
            kmm=i
            break
        Pgbest=Pgbest+1.0/len(archive_Cost)
    gbest=archive_Position[kmm]#从解集中选取一个值作为全局最优
    pbest=pop_Position#设定自身最优解

    for it in range(M):
        print it
        c=cmax-(cmax-cmin)*it/M
        for i in range(N):
            #if  i>19:
               # break
            #print i
            for j in range(D):
                #print i
                v[i][j]=v[i][j]+c*random.random()*(pbest[i][j]-pop_Position[i][j])+c*random.random()*(gbest[j]-pop_Position[i][j])
                if v[i][j]>vmax[j]:
                    v[i][j]=vmax[j]
                if v[i][j]<vmin[j]:
                    v[i][j]=vmin[j]
                pop_Position[i][j]=pop_Position[i][j]+v[i][j]
                if pop_Position[i][j]>ub[j]:
                    pop_Position[i][j]=lb[j]+(ub[j]-lb[j])*random.random()
                    v[i][j]=(ub[j]-lb[j])*random.random()*0.5
                if pop_Position[i][j]<lb[j]:
                    pop_Position[i][j] = lb[j] + (ub[j] - lb[j]) * random.random()
                    v[i][j] = (ub[j] - lb[j]) * random.random() * 0.5
                if Dominates(CostFunction(pop_Position[i]),CostFunction(pbest[i])):
                    pbest[i]=pop_Position[i]
                pop_Cost = []
                for ki in range(N):
                    pop_Cost.append(CostFunction(pop_Position[ki]))
        ndpop_Position = []  # clear the transfer matrix
        ndpop_Cost = []
        pop_IsDominated = DetermineDomination(pop_Cost)
        for ki in range(len(pop_Cost)):
            if pop_IsDominated[ki] == 0:
                ndpop_Position.append(pop_Position[ki])
                ndpop_Cost.append(pop_Cost[ki])
        for kt in range(len(ndpop_Cost)):
            archive_Position.append(ndpop_Position[kt])
            archive_Cost.append(ndpop_Cost[kt])
        ndpop_Position = []
        ndpop_Cost = []
        pop_IsDominated = DetermineDomination(archive_Cost)
        for ki in range(len(archive_Cost)):
            if pop_IsDominated[ki] == 0:
                ndpop_Position.append(archive_Position[ki])
                ndpop_Cost.append(archive_Cost[ki])
        archive_Cost = []
        archive_Position = []
        for kt in range(len(ndpop_Cost)):
            mt = 0
            for kj in range(len(archive_Cost)):
                # print len(archive_Cost)
                if kt != 1:
                    if archive_Cost[kj][0] == ndpop_Cost[kt][0]:
                        if archive_Cost[kj][1] == ndpop_Cost[kt][1]:
                            if archive_Cost[kj][2] == ndpop_Cost[kt][2]:
                                mt = 1
                                break
            if mt == 0:
                archive_Position.append(ndpop_Position[kt])
                archive_Cost.append(ndpop_Cost[kt])
            Pgbest = 1 / len(archive_Cost)
            kl = random.random()
            for Kt in range(len(archive_Cost)):
                if kl < Pgbest:
                    km = Kt
                    break
                Pgbest = Pgbest + 1 / len(archive_Cost)
            gbest = archive_Position[km]

            number_archive = len(archive_Position)
            while number_archive > Nup:
                k = select_delect(archive_Cost)
                archive_Cost[k] = 10000 * numpy.ones(3)
                number_archive = number_archive - 1
        # from mpl_toolkits.mplot3d import Axes3D
        # x = [t[0] for t in archive_Cost]
        # y = [t[1] for t in archive_Cost]
        # z = [t[2] for t in archive_Cost]
        # fig=plt.figure(11)
        # ax = Axes3D(fig)
        # ax.scatter(x,y , z, c='r')  # 绘点
        # #plt.plot(x, y, 'ro')
        # plt.xlabel('f_1')
        # plt.ylabel('f_2')
        # plt.show()
    total_archive_Cost.append(archive_Cost)
    total_archive_Position.append(archive_Position)
mydata = []
mydata = total_archive_Cost

thefile = open("DTLZ1_MOPSO_COST.txt", "w+")
for item in mydata:
    thefile.write("%s\n" % item)
thefile.close()
mydata = []
mydata1 = total_archive_Position
thefile = open("DTLZ1_MOPSO_Position.txt", "w+")
for item in mydata1:
    thefile.write("%s\n" % item)
thefile.close()

# x=[tk[0] for tk in archive_Cost]
# y=[tk[1] for tk in archive_Cost]
# print len(archive_Cost)
# #print y
#
# plt.figure(11)
# plt.plot(x,y, 'ro')
# plt.xlabel('f_1')
# plt.ylabel('f_2')
# plt.show()
























