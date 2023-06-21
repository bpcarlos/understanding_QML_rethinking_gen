#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import tensorcircuit as tc

def main(training_data):
    def convolutional_layer(c, q1, q2, param):
        c.r(q1, theta=param[0], alpha=param[1], phi=param[2])
        c.r(q2, theta=param[3], alpha=param[4], phi=param[5])
        c.exp1(q1, q2, theta=param[6], unitary=tc.gates._zz_matrix)
        c.r(q1, theta=param[7], alpha=param[8], phi=param[9])
        c.r(q2, theta=param[10], alpha=param[11], phi=param[12])
        return c
    
    def pooling_layer(c, q1, q2, param):
        c.cr(q1, q2, theta=param[0], alpha=param[1], phi=param[2])
        return c
    
    def accuracy(params):
        # compute accuracy
        accuracy_data = 0
        for j in range(len(gs_list)):
            c = tc.Circuit(nqubits, mps_inputs=gs_list[j])
            
            convolutional_layer(c, 0, 1, params[:13])
            convolutional_layer(c, 2, 3, params[:13])
            convolutional_layer(c, 4, 5, params[:13])
            convolutional_layer(c, 6, 7, params[:13])
            
            convolutional_layer(c, 1, 2, params[:13])
            convolutional_layer(c, 3, 4, params[:13])
            convolutional_layer(c, 5, 6, params[:13])
            convolutional_layer(c, 7, 0, params[:13])
            
            pooling_layer(c, 0, 1, params[13:16])
            pooling_layer(c, 2, 3, params[13:16])
            pooling_layer(c, 4, 5, params[13:16])
            pooling_layer(c, 6, 7, params[13:16])
            
            convolutional_layer(c, 1, 3, params[16:29])
            convolutional_layer(c, 5, 7, params[16:29])
            
            convolutional_layer(c, 3, 5, params[16:29])
            convolutional_layer(c, 7, 1, params[16:29])
            
            pooling_layer(c, 1, 3, params[29:32])
            pooling_layer(c, 5, 7, params[29:32])
            
            convolutional_layer(c, 3, 7, params[32:45])
            
            zz3 = tc.templates.measurements.mpo_expectation(c, mpo3)
            zz7 = tc.templates.measurements.mpo_expectation(c, mpo7)
            zz37 = tc.templates.measurements.mpo_expectation(c, mpo37)
            
            proj_00 = (1+zz37+zz3+zz7)/4
            proj_01 = (1-zz37-zz3+zz7)/4
            proj_10 = (1-zz37+zz3-zz7)/4
            proj_11 = (1+zz37-zz3-zz7)/4
            
            if label_list[j] == 0:
                if proj_00 < proj_01 and proj_00 < proj_10 and proj_00 < proj_11:
                    accuracy_data += 1
            elif label_list[j] == 1:
                if proj_01 < proj_00 and proj_01 < proj_10 and proj_01 < proj_11:
                    accuracy_data += 1
            elif label_list[j] == 2:
                if proj_10 < proj_00 and proj_10 < proj_01 and proj_10 < proj_11:
                    accuracy_data += 1
            else:
                if proj_11 < proj_00 and proj_11 < proj_01 and proj_11 < proj_10:
                    accuracy_data += 1
                

        return accuracy_data*100/len(gs_list)
        
    nqubits = 8
    best_params = np.loadtxt(f"{training_data}_training_data/BEST_PARAMS_j1j2_{training_data}")
    
    label_list = np.loadtxt(f"{training_data}_training_data/LABELS_{training_data}")    
    gs_list = np.load(f"{training_data}_training_data/train_groundstates.npy", allow_pickle=True)
        
    id0, id1, id2, z3, id4, id5, id6, id7 = tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.z(), tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.i()   
    mpo3 = tc.quantum.QuOperator([id0[0], id1[0], id2[0], z3[0], id4[0], id5[0], id6[0], id7[0]], [id0[1], id1[1], id2[1], z3[1], id4[1], id5[1], id6[1], id7[1]])

    id0, id1, id2, id3, id4, id5, id6, z7 = tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.z()   
    mpo7 = tc.quantum.QuOperator([id0[0], id1[0], id2[0], id3[0], id4[0], id5[0], id6[0], z7[0]], [id0[1], id1[1], id2[1], id3[1], id4[1], id5[1], id6[1], z7[1]])

    id0, id1, id2, z3, id4, id5, id6, z7 = tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.z(), tc.gates.i(), tc.gates.i(), tc.gates.i(), tc.gates.z()   
    mpo37 = tc.quantum.QuOperator([id0[0], id1[0], id2[0], z3[0], id4[0], id5[0], id6[0], z7[0]], [id0[1], id1[1], id2[1], z3[1], id4[1], id5[1], id6[1], z7[1]])

    train_accuracy = accuracy(best_params)
    print(train_accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default=5, type=int)
    args = parser.parse_args()
    main(**vars(args))
