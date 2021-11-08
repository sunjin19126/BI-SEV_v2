from set_ensemble import ensembleModelResults

Etype = 'Avg'
ValidResult = []
IndepResult = ['result/5/SEV/RFscore/SEV_Ite/RF.ite4.txt','result/5/SEV/CNNscore/SEV_Ite/CNN.ite4.txt','result/5/SEV/RNNscore/SEV_Ite/RNN.ite4.txt']
AvgSaveDir = 'result/5/Ensemble/Ens_Avg/Ens'
HWSaveDir = 'result/5/Ensemble/Ens_HW/Ens'
SWSaveDir = 'result/5/Ensemble/Ens_SW/Ens'
SWModelSaveDir = 'result/5/Ensemble/Ens_SW_Model/Ens'

ensembleModelResults(EnsembleType=Etype,
                     VResult_dirList=ValidResult,
                     IResult_dirList=IndepResult,
                     AvgSsave_dir=AvgSaveDir,
                     HWSsave_dir=HWSaveDir,
                     SWSsave_dir=SWSaveDir,
                     SWMsave_dir=SWModelSaveDir
                     )