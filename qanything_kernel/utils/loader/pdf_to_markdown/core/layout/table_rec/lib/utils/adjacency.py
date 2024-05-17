from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def adjacency (box1, box2):
    sr1 = box1[0]
    er1 = box1[1]
    sc1 = box1[2]
    ec1 = box1[3]
    
    sr2 = box2[0]
    er2 = box2[1] 
    sc2 = box2[2]
    ec2 = box2[3]
    
    if sr2 - er1 == 1 or sr1 - er2 == 1:
        #l-r adjacency
        if (sc1<=sc2 and ec1>=ec2) or (sc1>=sc2 and ec1<=ec2):
            return True
        else:
            return False
    elif sc2 - ec1 == 1 or sc1 - ec2 == 1:
        #u-d adjacency
        if (sr1<=sr2 and er1>=er2) or (sr1>=sr2 and er1<=er2):
            return True
        else:
            return False
    else:
        #no adjacency
        return False

def same_row(box1, box2):
    sc1 = box1[0]
    ec1 = box1[1]
    sr1 = box1[2]
    er1 = box1[3]
    
    sc2 = box2[0]
    ec2 = box2[1] 
    sr2 = box2[2]
    er2 = box2[3]

    #unit1.axis[2] sr1
    #unit1.axis[3] er1
    #unit2.axis[2] sr2
    #unit2.axis[3] er2

    if sr1 <= sr2 <= er1:
        return True

    elif sr2 <= sr1 <= er2:
        return True
        
    else:
        return False

def same_col(box1, box2):
    sc1 = box1[0]
    ec1 = box1[1]
    sr1 = box1[2]
    er1 = box1[3]
    
    sc2 = box2[0]
    ec2 = box2[1] 
    sr2 = box2[2]
    er2 = box2[3]

    #unit1.axis[0] sc1 
    #unit2.axis[0] sc2
    #unit1.axis[1] ec1
    #unit2.axis[1] ec2

    if sc1 <= sc2 <= ec1:
        return True
    elif sc2 <= sc1 <= ec2:
        return True
    else:
        return False


def loss_mask(num_obj):

    mask = torch.zeros(num_obj, num_obj)
    for i in range(num_obj):
        for j in range(i):
            mask[i][j] = 1.0

    return mask

def v_adjacency (box1, box2):
    sr1 = box1[0]
    er1 = box1[1]
    sc1 = box1[2]
    ec1 = box1[3]
    
    sr2 = box2[0]
    er2 = box2[1] 
    sc2 = box2[2]
    ec2 = box2[3]
    
    if sc2 - ec1 == 1 or sc1 - ec2 == 1:
        #u-d adjacency
        if (sr1<=sr2 and er1>=er2) or (sr1>=sr2 and er1<=er2):
            return True
        else:
            return False
    else:
        #no adjacency
        return False

def h_adjacency (box1, box2):
    sr1 = box1[0]
    er1 = box1[1]
    sc1 = box1[2]
    ec1 = box1[3]
    
    sr2 = box2[0]
    er2 = box2[1] 
    sc2 = box2[2]
    ec2 = box2[3]
    
    if sr2 - er1 == 1 or sr1 - er2 == 1:
        #l-r adjacency
        if (sc1<=sc2 and ec1>=ec2) or (sc1>=sc2 and ec1<=ec2):
            return True
        else:
            return False
    else:
        #no adjacency
        return False