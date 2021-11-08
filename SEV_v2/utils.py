import numpy as np
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import save_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.model_selection import StratifiedKFold

from Models import *

AA = 'ARNDCQEGHILKMFPSTWYV'

class Config(object):
    def __init__(self,
                 dataLoad_dir,
                 modelLoad_dir,
                 encoding_type,
                 model_type,
                 modelSave_dir,
                 ):

        self.dataLoad_dir = dataLoad_dir
        self.modelLoad_dir = modelLoad_dir
        self.encoding = encoding_type
        self.model = model_type

def getData(dataPath,dataType,seed=42):

    seqs = open(dataPath).readlines()
    if dataType =='train':
        np.random.seed(seed)
        np.random.shuffle(seqs)
    else:
        pass
    X = [[AA.index(res.upper()) if res.upper() in AA else 20 for res in (seq.split()[0])]
         for seq in seqs if seq.strip() != '']
    y = np.array([int(seq.split()[-1]) for seq in seqs if seq.strip() != ''])

    return X,y

# getData与trainModel之间,还有一个Feature Extraction的过程,但那些代码都包含在EncodingandModels.py文件中

def trainModelBySEV(modelKind,modelType,TrainSet,trainl,TestSet,testl,Msave_dir,VSsave_dir,ISsave_dir,TrainingEpochs):
    ''' 无论是SEV还是10折交叉验证，其整体处理过程如下
        Start：接收Features和Lables
        Then：切分数据，并训练模型。在模型的训练过程中，保存验证结果
        End:所有模型训练完以后，打印每个模型的验证结果，并将此结果作为判定模型效果的指标，打印并保存。'''
    X_train,X_val,y_train,y_val = create_examples_SE(TrainSet,trainl)
    X_ind,y_ind = TestSet,testl
    #AucList = []

    if modelType == 'ML':
        for i,_ in enumerate(X_train):
            # 这里需要将每份Xi和yi拿出90%训练，10%做验证
            X9,y9 = X_train[i],y_train[i],
            X1,y1 = X_val[i],y_val[i]

            if modelKind == 'RF':
                model = RFwe()
            else:
                print('No other ML models')
            model.fit(X9,y9)
            if Msave_dir:
                joblib.dump(model,Msave_dir + '.%d.pkl' % (i), compress=9)
            if VSsave_dir:
                #print('len(X1)' + str(len(X1)))
                print('X1_' + str(i) + '的shape:' + str(X1.shape))
                #print('len(y1)' + str(len(y1)))
                print('y1_' + str(i) + '的shape:' + str(y1.shape))
                vals = np.zeros((len(y1),2))
                print('vals' + str(i) + '的shape:' +str(vals.shape))
                vals[:,0] = y1
                vals[:,1] = model.predict_proba(X1)[:,1]
            np.savetxt(VSsave_dir + 'val%d.txt' % (i), vals, fmt='%f', delimiter='\t')
            print('基于SEV的ML模型' + str(i) + '的验证结果')
            assessment(vals[:,0],vals[:,-1])
            #AucList.append(V_auc)
            if ISsave_dir:
                inds = np.zeros((len(y_ind),2))
                inds[:,0] = y_ind
                inds[:,1] = model.predict_proba(X_ind)[:,1]
            np.savetxt(ISsave_dir + 'ind%d.txt' % (i), inds, fmt='%f', delimiter='\t')
            print('基于SEV的ML模型' + str(i) + '的独立测试结果')
            assessment(inds[:,0],inds[:,-1])


    elif modelType == 'DL':
        for i,Xi in enumerate(X_train):

            X9, y9 = X_train[i], y_train[i],
            X1, y1 = X_val[i], y_val[i]

            if modelKind == 'LSTM':
                net = LSTMwe()
            elif modelKind == 'CNN':
                net = CNN1Dwe()
            else:
                print('No other DL models')

            best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (Msave_dir,i),monitor='val_loss',
                                         verbose=1,save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=100)
            net.fit(X9,y9,batch_size=512,epochs=TrainingEpochs,verbose=2,callbacks=[best_saving,early_stopping])
            if Msave_dir:
                save_model(net,'%s.%d.h5' % (Msave_dir,i))
            if VSsave_dir:
                print('X1_' + str(i) + '的shape:' + str(X1.shape))
                # print('len(y1)' + str(len(y1)))
                print('y1_' + str(i) + '的shape:' + str(y1.shape))
                vals = np.zeros((len(X1), 2))
                print('vals' + str(i) + '的shape:' + str(vals.shape))
                vals[:,0] = y1
                vals[:,1] = net.predict_proba(X1,batch_size=512)[:,0]
            np.savetxt(VSsave_dir + 'val%d.txt' % (i), vals, fmt='%f', delimiter='\t')
            print('基于SEV的DL模型' + str(i) + '的验证结果')
            assessment(vals[:,0],vals[:,1])
            #AucList.append(V_auc)
            if ISsave_dir:
                # print('Independent Test:', net.evaluate(X_ind, y_ind, batch_size=512))
                inds = np.zeros((len(X_ind),2))
                inds[:,0] = y_ind
                inds[:,1] = net.predict_proba(X_ind,batch_size=512)[:,0]
            np.savetxt(ISsave_dir + 'ind%d.txt' % (i), inds, fmt='%f', delimiter='\t')
            print('基于SEV的DL模型' + str(i) + '的独立测试结果')
            assessment(inds[:,0],inds[:,1])


    else:
        print('modelType error !')

    #return AucList

def trainModelBy10kCV(modelKind,modelType,TrainSet,trainl,TestSet,testl,Msave_dir,VSsave_dir,ISsave_dir,TrainingEpochs):

    X,y = TrainSet,trainl
    folds = StratifiedKFold(10).split(X,y)
    X_ind,y_ind = TestSet,testl

    if modelType == 'ML':
        for i,(trained,valided) in enumerate(folds):
            X_train,y_train = X[trained],y[trained]
            X_valid,y_valid = X[valided],y[valided]

            if modelKind == 'RF':
                model = RFwe()
            else:
                print('No other ML models')
            model.fit(X_train,y_train)
            if Msave_dir:
                joblib.dump(model,Msave_dir + '.%d.pkl' % (i),compress=9)
            if VSsave_dir:
                vals = np.zeros((len(X_valid),2))
                vals[:,0] = y_valid
                vals[:,1] = model.predict_proba(X_valid)[:,1]
            np.savetxt(VSsave_dir+'val%d.txt' % (i), vals,fmt='%f', delimiter='\t')
            print('基于10k-CV的ML模型' + str(i) + '的验证结果')
            assessment(vals[:,0],vals[:,1])
            if ISsave_dir:
                inds = np.zeros((len(X_ind),2))
                inds[:,0] = y_ind
                inds[:,1] = model.predict_proba(X_ind)[:,1]
            np.savetxt(ISsave_dir + 'ind%d.txt' % (i), inds, fmt='%f', delimiter='\t')
            print('基于10k-CV的ML模型' + str(i) + '的独立测试结果')
            assessment(inds[:,0],inds[:,1])

    elif modelType == 'DL':
        for i,(trained,valided) in enumerate(folds):
            X_train,y_train = X[trained],y[trained]
            X_valid,y_valid = X[valided],y[valided]

            if modelKind == 'LSTM':
                net = LSTMwe()
            elif modelKind == 'CNN':
                net = CNN1Dwe()
            else:
                print('No other DL models')

            best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (Msave_dir, i), monitor='val_loss',
                                          verbose=1, save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=100)
            net.fit(X_train, y_train, batch_size=512, epochs=TrainingEpochs, verbose=2, callbacks=[best_saving, early_stopping])

            if Msave_dir:
                save_model(net,'%s.%d.h5' % (Msave_dir,i))
            if VSsave_dir:
                vals = np.zeros((len(X_valid),2))
                vals[:,0] = y_valid
                vals[:,1] = net.predict_proba(X_valid,batch_size=512)[:,0]
            np.savetxt(VSsave_dir + 'val%d.txt' % (i), vals, fmt='%f', delimiter='\t')
            print('基于10k-CV的DL模型' + str(i) + '的验证结果')
            assessment(vals[:,0],vals[:,1])
            if ISsave_dir:
                inds = np.zeros((len(X_ind),2))
                inds[:,0] = y_ind
                inds[:,1] = net.predict_proba(X_ind,batch_size=512)[:,0]
            np.savetxt(ISsave_dir + 'ind%d.txt' % (i), inds, fmt='%f', delimiter='\t')
            print('基于10k-CV的DL模型' + str(i) + '的独立测试结果')
            assessment(inds[:,0],inds[:,1])

    else:
        print('modelType error !')

def assessment(labels,scores):
    # 这里最好再加上一些参数，可以更准确地评价结果是Val or Test?以及是哪个模型的结果
    y_pred = [int(item >0.5) for item in scores]

    auc = metrics.roc_auc_score(labels,scores)
    acc = accuracy_score(labels,y_pred)

    mcc = matthews_corrcoef(labels,y_pred)

    tn,fp,fn,tp = confusion_matrix(labels,y_pred).ravel()
    sn = float(tp)/float(tp+fn)
    sp = float(tn)/float(tn+fp)

    print('AUC: %f' % auc)
    print('Acc: %f' % acc)
    print('MCC: %f' % mcc)
    print('Sn: %f' % sn)
    print('Sp: %f' % sp)

    return auc
# 这里就不要再单独写test函数了,这跟BERT那个代码还不一样.
# 那套代码中，训练的过程只产生一个模型，而且那个模型是一直贯穿全过程的.
# 这里不一样,训练过程产生多个基础模型,如果单独写testModel函数,需要先保存后调用,又慢又麻烦.
# 另外还有一点,明天写代码之前,先想好各个变量怎么命名,尤其是测试集的X和y,建议命名X_ind,y_ind

'''----------------------------------分割线-------------------------------------------'''
def create_examples_SE(Lines,y):

    # Step1:将不同类的样本区分开来
    Examples_s,Labels_s,Examples_l,Labels_l = [],[],[],[]
    for (i,line) in enumerate(Lines):
        if y[i] == 1:
            Examples_s.append(line)
            Labels_s.append(y[i])
        elif y[i] == 0:
            Examples_l.append(line)
            Labels_l.append(y[i])
        else:
            print("Data" + str(i) + "Split Failed")

    # Step2:计算两类样本比例,并根据少数类样本数量来切分多数类样本,获取多数类样本对应的索引号
    len_l,len_s = len(Examples_l),len(Examples_s)
    ratio = round(len_l/len_s)
    Idx = []
    idx = 0
    for _ in range(ratio):
        Idx.append(idx)
        idx += len_s

    # Step3:将少数类样本与切分好的若干份多数样本子集进行组合,得到若干样本均衡的子数据集;
    # 同时,在此过程中,还对生成的均衡子样本集进行数据划分,90%做训练,10%做验证
    ExamplesT,LabelsT,ExamplesV,LabelsV = [],[],[],[]
    for i,id in enumerate(Idx):
        ExampleT,ExampleV = [],[]
        LabelT,LabelV = [],[]
        for example_t in Examples_s[0 : int(len_s*0.9)]:
            ExampleT.append(example_t)
        for example_v in Examples_s[int(len_s*0.9) : len_s]:
            ExampleV.append(example_v)
        for example_t in Examples_l[id : int(id+(len_s)*0.9)]:
            ExampleT.append(example_t)
        for example_v in Examples_l[int(id+(len_s)*0.9) : (id+len_s)]:
            ExampleV.append(example_v)

        for label_t in Labels_s[0 : int(len_s*0.9)]:
            LabelT.append(label_t)
        for label_v in Labels_s[int(len_s*0.9) : len_s]:
            LabelV.append(label_v)
        for label_t in Labels_l[id : int(id+(len_s)*0.9)]:
            LabelT.append(label_t)
        for label_v in Labels_l[int(id+(len_s)*0.9) : (id+len_s)]:
            LabelV.append(label_v)
        ExamplesT.append(ExampleT)
        ExamplesV.append(ExampleV)
        LabelsT.append(LabelT)
        LabelsV.append(LabelV)

    return np.array(ExamplesT), np.array(ExamplesV), np.array(LabelsT), np.array(LabelsV)

def getAucList(path_list):

    AucList = []
    for i,path in enumerate(path_list):
        seqs = open(path).readlines()
        y_test = np.array([float(seq.split()[0]) for seq in seqs if seq.strip() != ''])
        y_proba = np.array([float(seq.split()[-1]) for seq in seqs if seq.strip() != ''])
        auc = metrics.roc_auc_score(y_test, y_proba)
        AucList.append(auc)
    return AucList