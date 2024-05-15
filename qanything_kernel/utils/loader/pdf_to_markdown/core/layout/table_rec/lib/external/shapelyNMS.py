import numpy as np
from shapely.geometry import *

def delet_min_first(dets,pts,areas,inter_areas,min_areas,scores,thresh_min,thresh_conf):
    choose_dict = {}
    for i in range(0,len(pts)):
        areai = areas[i]
        for j in range(0, len(pts)):
            if i==j:
                continue
            areaj = areas[j]
            ovr = inter_areas[i][j] / min_areas[i][j]
            if ovr > thresh_min and areai != min_areas[i][j]:
                if str(i) not in choose_dict.keys():
                    choose_dict[str(i)] = [j]
                else:
                    choose_dict[str(i)].append(j)
    delet_list = []
    for i in choose_dict.keys():
        ovr_list = choose_dict[i]
        if len(ovr_list) == 1:
            j = choose_dict[i][0]
            #ovr_max = inter_areas[int(i)][j] / max(areas[int(i)],areas[j])
            #ovr_min = inter_areas[int(i)][j] / min(areas[int(i)],areas[j])
            #if ovr_min > thresh_min:
            #    conf_i,conf_j = scores[int(i)],scores[j]
            #    #if abs(conf_i - conf_j) > thresh_conf:
            #    #    index = int(i) if conf_i < conf_j else j
            #    #    delet_list.append(index)
            #    #else:
            delet_list.append(j)
        elif len(ovr_list) == 2:
            j,k = choose_dict[i][0],choose_dict[i][1]
            area_sum = areas[j] + areas[k] - inter_areas[j][k]
            ovr = area_sum / areas[int(i)]
            if ovr > thresh_min and ovr < 1.1:
                delet_list.append(j)
                delet_list.append(k)
        else:
            for index in choose_dict[i]:
                delet_list.append(index)
    dets = list(dets)
    keep = []
    for i in range(len(dets)):
        if i not in delet_list:
            keep.append(dets[i])
    return np.array(keep)#dets

def delet_min(dets,pts,areas,inter_areas,min_areas,scores,thresh_min,thresh_conf):
    choose_dict = {}
    for i in range(0,len(pts)):
        areai = areas[i]
        for j in range(0, len(pts)):
            if i==j:
                continue
            areaj = areas[j]
            ovr = inter_areas[i][j] / (areas[i] + areas[j] - inter_areas[i][j])
            if ovr > thresh_min and areai != min_areas[i][j]:
                if str(i) not in choose_dict.keys():
                    choose_dict[str(i)] = [j]
                else:
                    choose_dict[str(i)].append(j)
    delet_list = []
    for i in choose_dict.keys():
        ovr_list = choose_dict[i]
        if len(ovr_list) == 1:
            j = choose_dict[i][0]
            delet_list.append(j)
        elif len(ovr_list) == 2:
            j,k = choose_dict[i][0],choose_dict[i][1]
            area_sum = areas[j] + areas[k] - inter_areas[j][k]
            ovr = area_sum / areas[int(i)]
            if ovr > thresh_min and ovr < 1.1:
                delet_list.append(j)
                delet_list.append(k)
        '''
        else:
            for index in choose_dict[i]:
                delet_list.append(index)
        '''
    dets = list(dets)
    keep = []
    for i in range(len(dets)):
        if i not in delet_list:
            keep.append(dets[i])
    return np.array(keep)#dets


def cross_union(dets,pts,areas,inter_areas,min_areas,scores,thresh_min,thresh_conf):
    choose_dict = {}
    for i in range(0,len(pts)):
        choose_dict[i] = 0
    
    for i in range(0,len(pts)):
        areai = areas[i]
        for j in range(0, len(pts)):
            if i==j or scores[i] < scores[j]:
                continue
            areaj = areas[j]
            ovr_union = inter_areas[i][j] / (areai+areaj-inter_areas[i][j])
            ovr_min = inter_areas[i][j] / min(areai,areaj)
            if ovr_union > thresh_min:
                #if scores[i]-scores[j]>0.2:
                #    choose_dict[j]=1
                #elif scores[j]-scores[i]>0.2:
                #    choose_dict[i]=1
                if areai > areaj:
                    choose_dict[j]=1
                else:
                    choose_dict[i]=1
    dets = list(dets)
    keep = []
    for i in range(len(dets)):
        if choose_dict[i]==0:
            keep.append(dets[i])
    return np.array(keep)#dets
   

def pnms(dets,thresh_min,thresh_conf):
    scores = dets[:,-1]

    pts = []
    for i in range(dets.shape[0]):
        pts.append([dets[i][0:2],dets[i][2:4],dets[i][4:6],dets[i][6:8]])

    areas = np.zeros(scores.shape)
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0],scores.shape[0]))
    min_areas = np.zeros((scores.shape[0],scores.shape[0]))
    
    for i in range(0,len(pts)):
        polyi = Polygon(pts[i])
        areas[i] = polyi.area
    
        for j in range(i, len(pts)):
            polyj = Polygon(pts[j])
            try:
                inS = polyi.intersection(polyj)
            except:
                print(pts[i])
                print(pts[j]) 
            inter_areas[i][j] = inS.area
            inter_areas[j][i] = inS.area
            min_areas[i][j] = min(areas[i],polyj.area)
            min_areas[j][i] = min_areas[i][j]

    keep = []
    '''
    while order.size > 0:
        i = order[0]
        keep.append(dets[i])
        #ovr = inter_areas[i][order[1:]] / min_areas[i][order[1:]]
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    '''
    #return delet_min(dets,pts,areas,inter_areas,min_areas,scores,thresh_min,thresh_conf)
    return cross_union(dets,pts,areas,inter_areas,min_areas,scores,thresh_min,thresh_conf)
