from collections import defaultdict
import openpyxl
from collections import defaultdict
import pickle
import os
import numpy as np
from six.moves import cPickle


def accumulate_values(col, start):
    tmp = []
    for i in range(3):
        if col[start+i].value != None:
            tmp.append(col[start+i].value)
    return tmp

def fill_excel_train(col, start_row, data):
    col[start_row].value = float(data["IS"])
    col[start_row+4].value = data["FID"]
    col[start_row+8].value = data["Improved_Precision"]
    col[start_row+12].value = data["Improved_Recall"]
    col[start_row+16].value = data["Density"]
    col[start_row+20].value = data["Coverage"]
    col[start_row+24].value = f'{data["Top1_acc"]}/{data["Top5_acc"]}'
    
def fill_excel_valid(col, start_row, data):
    col[start_row].value = data["FID"]
def fill_excel_ifid(col, start_row, data):
    col[start_row].value = sum(data)/len(data)
def fill_excel_extra(col):
    pairs = [(10, 107), (42,134), (74,161)]
    for pair in pairs:
        for i in range(8):
            tmp = accumulate_values(col, pair[0]+4*i)
            if i == 6:
                top1 = []
                top5 = []
                for t in tmp:
                    top1.append(float(t[:t.find("/")]))
                    top5.append(float(t[t.find("/")+1:]))
                if len(tmp) == 0:
                    col[pair[1]+3*i].value = None
                    col[pair[1]+3*i+1].value = None
                elif len(tmp) == 1:
                    col[pair[1]+3*i].value = f"{round(np.mean(top1),2)}/{round(np.mean(top5),2)}"
                    col[pair[1]+3*i+1].value = "*"
                else:
                    col[pair[1]+3*i].value = f"{round(np.mean(top1),2)}/{round(np.mean(top5),2)}"
                    col[pair[1]+3*i+1].value = f"{round(np.std(top1, ddof=0),3)}/{round(np.std(top5, ddof=0),3)}"
                
            else:
                if len(tmp) == 0:
                    col[pair[1]+3*i].value = None
                    col[pair[1]+3*i+1].value = None
                elif len(tmp) == 1:
                    col[pair[1]+3*i].value = round(np.mean(tmp),2)
                    col[pair[1]+3*i+1].value = "*"
                else:
                    col[pair[1]+3*i].value = round(np.mean(tmp),2)
                    col[pair[1]+3*i+1].value = round(np.std(tmp, ddof=0),3)

def check_SN(name):
    if name.startswith("CIFAR10-SNGAN-Diff"):
        return False
    if name.startswith("CIFAR10-SNGAN-ADA"):
        return False
    if name.startswith("CIFAR10-SNGAN-APA"):
        return False
    if name.startswith("CIFAR10-SNGAN-LeCam"):
        return False
    return True
    
def get_value(col, row):
    if col[row].value != None:
        return col[row].value
    else:
        return 0
    
def get_pickle_names(col):
    pickle_names = defaultdict(dict)
    run_name1, run_name2, run_name3 = col[6].value, None, None
    if not run_name1.startswith("CIFAR10_") and check_SN(run_name1):
        backbones = ["InceptionV3_tf", "SwAV_torch", "Swin-T_torch"]
    else:
        backbones = ["InceptionV3_tf"]

    valid = "test" if run_name1.startswith("CIFAR") else "valid"
    if col[7].value != None:
        run_name2 = col[7].value
    if col[8].value != None:
        run_name3 = col[8].value
    for i, run_name in enumerate([run_name1, run_name2, run_name3]):
        if run_name != None:
            for backbone in backbones:
                pickle_names[f"run_name{i+1}"][f"{backbone}-train"] = run_name[run_name.find("-")+1:]+f"-{backbone}-train.pickle"
                # if not run_name.startswith("CIFAR10_") and check_SN(run_name):
                #     pickle_names[f"run_name{i+1}"][f"{backbone}-ifid"] = run_name+f"-{backbone}-train-ifid.pickle"
                #     if not run_name.startswith("AFHQ_V2_uncond"):
                #         pickle_names[f"run_name{i+1}"][f"{backbone}-valid"] = run_name+f"-{backbone}-{valid}.pickle"
    return pickle_names




wb = openpyxl.load_workbook("Taxonomy_experiments_43.xlsx")
pickles_path = "./eval_pickles/"
pickles_list = os.listdir(pickles_path)
# ws_list = ["Efficient-AFHQ-V2-256"]
ws_list = ["ImageNet_tailored", "Baby_ImageNet_tailored", "Papa_ImageNet_tailored", "Grandpa_ImageNet_tailored"]
IS_start_row = 10
SwAV_start_row = 42
SwinT_start_row = 74

for ws in ws_list:
    ws = wb[ws]
    all_columns = ws.columns
    
    for idx, col in enumerate(all_columns):        
        if idx >=2 and col[1].value != None:
            pickle_names = get_pickle_names(col)
            
            for i, key in enumerate(pickle_names.keys()): 
                # i, key = run_name1,2,3
                for key, value in pickle_names[key].items():
                    # key = "backbone-train/valid/ifid", value = "~~.pickle"
                    
                    # get starting row
                    if key.startswith("Inc"):
                        start_row = IS_start_row
                    elif key.startswith("SwAV"):
                        start_row = SwAV_start_row
                    else:
                        start_row = SwinT_start_row
                        
                    with open(os.path.join(pickles_path, value), "rb") as f:
                        data = pickle.load(f)
                        if key.endswith("train"):
                            fill_excel_train(col, start_row + i, data)
                        if key.endswith("valid"):
                            fill_excel_valid(col, start_row + i + 24, data)
                        if key.endswith("ifid"):
                            fill_excel_ifid(col, start_row + i + 28, data)
    
    # fill average and stds
    all_columns = ws.columns
    fid_inc, fid_swav, fid_swin = [], [], []
    top1_inc, top1_swav, top1_swin = [], [], []
    top5_inc, top5_swav, top5_swin = [], [], []
    is_inc, is_swav, is_swin = [], [], []
    ifid_inc, ifid_swav, ifid_swin = [], [], []
    
    for idx, col in enumerate(all_columns):
        if idx >= 2 and col[1].value != None:
            fill_excel_extra(col)
            
            for idx, backbone in enumerate([is_inc, is_swav, is_swin]):
                if col[107+27*idx].value != None:
                    backbone.append(col[107+27*idx].value)  
                    
            for idx, backbone in enumerate([fid_inc, fid_swav, fid_swin]):
                if col[110+27*idx].value != None:
                    backbone.append(col[110+27*idx].value)  
                    
            for idx, backbone in enumerate([top1_inc, top1_swav, top1_swin]):
                if col[125+27*idx].value != None:
                    backbone.append(float(col[125+27*idx].value[:col[125+27*idx].value.find("/")]))

            for idx, backbone in enumerate([top5_inc, top5_swav, top5_swin]):
                if col[125+27*idx].value != None:
                    backbone.append(float(col[125+27*idx].value[col[125+27*idx].value.find("/")+1:]))

            for idx, backbone in enumerate([ifid_inc, ifid_swav, ifid_swin]):
                if col[128+27*idx].value != None:
                    backbone.append(col[128+27*idx].value)            
                
    
    # fill fid rank
    all_columns = ws.columns
    for idx, col in enumerate(all_columns):
        if idx >= 2 and col[1].value != None:
            for idx, backbone in enumerate([is_inc, is_swav, is_swin]):
                if col[107+27*idx].value != None:
                    col[131+27*idx].value = sorted(backbone, reverse=True).index(col[107+27*idx].value)+1

            for idx, backbone in enumerate([fid_inc, fid_swav, fid_swin]):
                if col[110+27*idx].value != None:
                    col[132+27*idx].value = sorted(backbone).index(col[110+27*idx].value)+1
                    
            for idx, backbone in enumerate([top1_inc, top1_swav, top1_swin]):
                if col[125+27*idx].value != None:
                    col[133+27*idx].value = sorted(backbone, reverse=True).index(float(col[125+27*idx].value[:col[125+27*idx].value.find("/")]))+1
            
            
            for idx, backbone in enumerate([top5_inc, top5_swav, top5_swin]):
                if col[125+27*idx].value != None:
                    col[133+27*idx].value = f'{col[133+27*idx].value} / {sorted(backbone, reverse=True).index(float(col[125+27*idx].value[col[125+27*idx].value.find("/")+1:]))+1}'

            for idx, backbone in enumerate([ifid_inc, ifid_swav, ifid_swin]):
                if col[128+27*idx].value != None:
                    col[133+27*idx].value = f"{col[133+27*idx].value} / {sorted(backbone).index(col[128+27*idx].value)+1}"

wb.save("Taxonomy_experiments_43.xlsx")