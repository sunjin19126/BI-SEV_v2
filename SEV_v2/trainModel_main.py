from Encodings import *
from utils import *
from set_ensemble import *
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

c_modelType = 'DL'
c_model = 'LSTM'
c_Encoding = 'Index'
c_Epoch = 300
c_seed = 46

c_trainPath = 'dataset/chen_train.txt'
c_testPath = 'dataset/chen_test.txt'
c_Msave_dir = 'result/5/SEV/CNNmodel/CNN'
c_VSave_dir = 'result/5/SEV/CNNscore/SEV_Val/CNN'
c_ISave_dir = 'result/5/SEV/CNNscore/SEV_Ind/CNN'
c_IteSave_dir = 'result/5/SEV/CNNscore/SEV_Ite/CNN'
validSDL = ['result/5/SEV/CNNscore/SEV_Val/CNNval0.txt','result/5/SEV/CNNscore/SEV_Val/CNNval1.txt',
            'result/5/SEV/CNNscore/SEV_Val/CNNval2.txt','result/5/SEV/CNNscore/SEV_Val/CNNval3.txt',
            'result/5/SEV/CNNscore/SEV_Val/CNNval4.txt','result/5/SEV/CNNscore/SEV_Val/CNNval5.txt',
            'result/5/SEV/CNNscore/SEV_Val/CNNval6.txt','result/5/SEV/CNNscore/SEV_Val/CNNval7.txt',
            'result/5/SEV/CNNscore/SEV_Val/CNNval8.txt','result/5/SEV/CNNscore/SEV_Val/CNNval9.txt',
            'result/5/SEV/CNNscore/SEV_Val/CNNval10.txt','result/5/SEV/CNNscore/SEV_Val/CNNval11.txt'
            ]
indepSDL = ['result/5/SEV/CNNscore/SEV_Ind/CNNind0.txt','result/5/SEV/CNNscore/SEV_Ind/CNNind1.txt',
            'result/5/SEV/CNNscore/SEV_Ind/CNNind2.txt','result/5/SEV/CNNscore/SEV_Ind/CNNind3.txt',
            'result/5/SEV/CNNscore/SEV_Ind/CNNind4.txt','result/5/SEV/CNNscore/SEV_Ind/CNNind5.txt',
            'result/5/SEV/CNNscore/SEV_Ind/CNNind6.txt','result/5/SEV/CNNscore/SEV_Ind/CNNind7.txt',
            'result/5/SEV/CNNscore/SEV_Ind/CNNind8.txt','result/5/SEV/CNNscore/SEV_Ind/CNNind9.txt',
            'result/5/SEV/CNNscore/SEV_Ind/CNNind10.txt','result/5/SEV/CNNscore/SEV_Ind/CNNind11.txt'
            ]
'''
# Load Data
trainData,trainLabel = getData(c_trainPath,dataType='train',seed=c_seed)
testData,testLabel = getData(c_testPath,dataType='test')

# Feature Extraction
if c_Encoding == 'EAAC':
    trainFeatures = EAAC(trainData)
    testFeatures = EAAC(testData)
elif c_Encoding == 'EAAC2d':
    trainFeatures = EAAC2d(trainData)
    testFeatures = EAAC2d(testData)
elif c_Encoding == 'Index':
    trainFeatures = Index(trainData)
    testFeatures = Index(testData)

print('------------------------计时开始---------------------------------')
start_time = datetime.datetime.now()
T1 = datetime.datetime.strftime(start_time, '%Y-%m-%d %H:%M:%S')
print(str(c_model) + ' 训练开始时间：' + T1 )

# Training Models
trainModelBySEV(modelKind=c_model,
                modelType=c_modelType,
                TrainSet=trainFeatures,
                TestSet=testFeatures,
                trainl=trainLabel,
                testl=testLabel,
                Msave_dir=c_Msave_dir,
                VSsave_dir=c_VSave_dir,
                ISsave_dir=c_ISave_dir,
                TrainingEpochs = c_Epoch)

# trainModelBy10kCV(modelKind=c_model,
#                 modelType=c_modelType,
#                 TrainSet=trainFeatures,
#                 TestSet=testFeatures,
#                 trainl=trainLabel,
#                 testl=testLabel,
#                 Msave_dir=c_Msave_dir,
#                 VSsave_dir=c_VSave_dir,
#                 ISsave_dir=c_ISave_dir,
#                 TrainingEpochs = c_Epoch)

'''
# Validation & Integration
Auclist_V = getAucList(validSDL)
Auclist_I = getAucList(indepSDL)
n = len(Auclist_V)

for i in range(1,n+1):
    print('第' + str(i) + '次验证')
    concatValidScores(Auclist_V,validSDL, x=i)

end_time = datetime.datetime.now()
T2 = datetime.datetime.strftime(end_time, '%Y-%m-%d %H:%M:%S')
print(str(c_model) + ' 验证结束时间：' + T2 )
print('------------------------计时结束---------------------------------')

for i in range(1,n+1):
    print('第' + str(i) + '次测试')
    integrateIndepScores(Auclist_V, Auclist_I, indepSDL, c_IteSave_dir, x=i)

# Ensemble Results of different Models
