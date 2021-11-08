import numpy as np
from utils import assessment
from Models import MonoNN
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import save_model

def concatValidScores(AUCList_v,VScore_dirList,x):

    npAUCList_v = np.array(AUCList_v)
    list_v = npAUCList_v.argsort()[::-1]
    if x == 1:
        print('所有模型的验证得分从高到低排序如下：\n' + str(list_v))

    for i,l in enumerate(list_v[0:x]):
        VS_i = open(VScore_dirList[l]).readlines()
        VP_i = np.array([float(seq.split()[-1]) for seq in VS_i if seq.strip() != ''])
        VL_i = np.array([float(seq.split()[0]) for seq in VS_i if seq.strip() != ''])
        if i == 0:
            VP = VP_i
            VL = VL_i
        else:
            VP = np.r_[VP,VP_i]
            VL = np.r_[VL,VL_i]

    print('-----------由' + str(x) + '个Valid Scores Concat以后的结果：-----------')
    assessment(VL, VP)
    # if CctSave_dir:
    #     ite = np.zeros((len(VL), 2))
    #     ite[:, 0] = VL
    #     ite[:, 1] = VP
    #     np.savetxt(CctSave_dir + '.cct%d.txt' % (i), ite, fmt='%f', delimiter='\t')

def integrateIndepScores(AUCList_v,AUCList_i,IScore_dirList,IteSave_dir,x):

    npAUCList_v = np.array(AUCList_v)
    list_v = npAUCList_v.argsort()[::-1]
    # print('所有模型的验证得分从高到底排序如下：\n' + str(list_v))
    npAUCList_i = np.array(AUCList_i)
    list_i = npAUCList_i.argsort()[::-1]
    # print('所有模型的独立测试得分从高到底排序如下：\n' + str(list_i))

    if x == 1:
        print('所有模型的验证得分从高到底排序如下：\n' + str(list_v))
        print('所有模型的独立测试得分从高到底排序如下：\n' + str(list_i))

    for i, l in enumerate(list_v[0:x]):
        IS_i = open(IScore_dirList[l]).readlines()
        IP_i = np.array([float(seq.split()[-1]) for seq in IS_i if seq.strip() != ''])
        IL = np.array([float(seq.split()[0]) for seq in IS_i if seq.strip() != ''])
        if i == 0:
            IP = IP_i
        else:
            IP = IP + IP_i
    IP = IP / x

    print('-----------由' + str(x) + '个Indep Scores Integratie以后的结果：-----------')
    assessment(IL, IP)
    # if IteSave_dir:
    #     ite = np.zeros((len(IL), 2))
    #     ite[:, 0] = IL
    #     ite[:, 1] = IP
    #     np.savetxt(IteSave_dir + '.ite%d.txt' % (i), ite, fmt='%f', delimiter='\t')

def ensembleModelResults(EnsembleType,
                         VResult_dirList,IResult_dirList,
                         AvgSsave_dir,HWSsave_dir,SWSsave_dir,SWMsave_dir):
    # for i, Dir in enumerate(IResult_dirList):
    #     IR = open(Dir).readlines()
    #     IRL = np.array([float(seq.split()[0]) for seq in IR if seq.strip() != ''])
    #     if i == 0:
    #         IRP = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
    #     else:
    #         IRP_i = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
    #         IRP = np.c_[IRP, IRP_i]

    if EnsembleType == 'Avg':
        for i, Dir in enumerate(IResult_dirList):
            IR = open(Dir).readlines()
            IRL = np.array([float(seq.split()[0]) for seq in IR if seq.strip() != ''])
            if i == 0:
                IRP = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
            else:
                IRP_i = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
                IRP = IRP + IRP_i
        IRP = IRP / len(IResult_dirList)
        assessment(IRL,IRP)
        if AvgSsave_dir:
            avg = np.zeros((len(IRL),2))
            avg[:,0] = IRL
            avg[:,1] = IRP
        np.savetxt(AvgSsave_dir + '.avg.txt', avg, fmt='%f', delimiter='\t')

    elif EnsembleType == 'HW':
        for i, Dir in enumerate(IResult_dirList):
            IR = open(Dir).readlines()
            IRL = np.array([float(seq.split()[0]) for seq in IR if seq.strip() != ''])
            if i == 0:
                IRP0 = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
            elif i == 1:
                IRP1 = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
            elif i == 2:
                IRP2 = np.array([float(seq.split()[-1]) for seq in IR if seq.strip() != ''])
            else:
                print('No other HW Score files')
        IRP = (IRP0*0.8 + IRP1*0.7 + IRP2) / 2.5
        # IRP = (IRP0 * 0.7  + IRP1 * 1) / 1.7
        assessment(IRL, IRP)
        if HWSsave_dir:
            hws = np.zeros((len(IRL), 2))
            hws[:, 0] = IRL
            hws[:, 1] = IRP
       #np.savetxt(HWSsave_dir + '.hws.txt', hws, fmt='%f', delimiter='\t')

    elif EnsembleType == 'SW':

        for i,Dir in enumerate(VResult_dirList):
            VR = open(Dir).readlines()
            VRL = np.array([float(seq.split()[0]) for seq in VR if seq.strip() != ''])
            if i == 0:
                VRP = np.array([float(seq.split()[-1]) for seq in VR if seq.strip() != ''])
            else:
                VRP_i = np.array([float(seq.split()[-1]) for seq in VR if seq.strip() != ''])
                VRP = np.c_[VRP,VRP_i]

        for i, Dir in enumerate(IResult_dirList):
            SWR = open(Dir).readlines()
            SWRL = np.array([float(seq.split()[0]) for seq in SWR if seq.strip() != ''])
            if i == 0:
                SWRP = np.array([float(seq.split()[-1]) for seq in SWR if seq.strip() != ''])
            else:
                SWRP_i = np.array([float(seq.split()[-1]) for seq in SWR if seq.strip() != ''])
                SWRP = np.c_[SWRP, SWRP_i]

        net = MonoNN()
        best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (SWMsave_dir, i), monitor='val_loss',
                                      verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        net.fit(VRP, VRL, batch_size=128, epochs=10, verbose=2, callbacks=[best_saving, early_stopping])
        if SWMsave_dir:
            save_model(net, '%s.h5' % (SWMsave_dir))
        if SWSsave_dir:
            # print('swependent Test:', net.evaluate(X_sw, y_sw, batch_size=512))
            sws = np.zeros((len(SWRP), 2))
            sws[:, 0] = SWRL
            sws[:, 1] = net.predict_proba(SWRP, batch_size=512)[:, 0]
        np.savetxt(SWSsave_dir + 'sw.txt' , sws, fmt='%f', delimiter='\t')
        print('基于SEV的DL模型' + str(i) + '的独立测试结果')
        assessment(sws[:, 0], sws[:, 1])