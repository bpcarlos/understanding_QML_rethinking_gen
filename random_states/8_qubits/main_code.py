#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cma
from qibo.symbols import Z, I
from qibo import hamiltonians
from qibo import models, gates
import qibo
qibo.set_backend("numpy")

def main(training_data, accuracy_training):   
    def MPO_3():
        symbolic_expr = Z(3)*I(7)
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        return hamiltonian
        
    def MPO_7():
        symbolic_expr = Z(7)
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        return hamiltonian
        
        
    def MPO_37():
        symbolic_expr = Z(3)*Z(7)
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        return hamiltonian
    
    def convolutional_layer(c, q1, q2, param):
        c.add(gates.U3(q1, theta=param[0], phi=param[1], lam=param[2]))
        c.add(gates.U3(q1, theta=param[3], phi=param[4], lam=param[5]))
        c.add(gates.CU1(q1, q2, theta=param[6]))
        c.add(gates.U3(q1, theta=param[7], phi=param[8], lam=param[9]))
        c.add(gates.U3(q1, theta=param[10], phi=param[11], lam=param[12]))
        return c
    
    def pooling_layer(c, q1, q2, param):
        c.add(gates.CU3(q1, q2, theta=param[0], phi=param[1], lam=param[2]))
        return c
    
    def loss(params):
        cost = 0
        circuit = models.Circuit(nqubits)
        
        convolutional_layer(circuit, 0, 1, params[:13])
        convolutional_layer(circuit, 2, 3, params[:13])
        convolutional_layer(circuit, 4, 5, params[:13])
        convolutional_layer(circuit, 6, 7, params[:13])
        
        convolutional_layer(circuit, 1, 2, params[:13])
        convolutional_layer(circuit, 3, 4, params[:13])
        convolutional_layer(circuit, 5, 6, params[:13])
        convolutional_layer(circuit, 7, 0, params[:13])
        
        pooling_layer(circuit, 0, 1, params[13:16])
        pooling_layer(circuit, 2, 3, params[13:16])
        pooling_layer(circuit, 4, 5, params[13:16])
        pooling_layer(circuit, 6, 7, params[13:16])
        
        convolutional_layer(circuit, 1, 3, params[16:29])
        convolutional_layer(circuit, 5, 7, params[16:29])
        
        convolutional_layer(circuit, 3, 5, params[16:29])
        convolutional_layer(circuit, 7, 1, params[16:29])
        
        pooling_layer(circuit, 1, 3, params[29:32])
        pooling_layer(circuit, 5, 7, params[29:32])
        
        convolutional_layer(circuit, 3, 7, params[32:45])
        
        for j in range(training_data):
            final_state = circuit(gs_list[j]).state()

            z3 = np.real(ham3.expectation(final_state))
            z7 = np.real(ham7.expectation(final_state))
            zz37 = np.real(ham37.expectation(final_state))
            
            proj_00 = (1+zz37+z3+z7)/4
            proj_01 = (1-zz37-z3+z7)/4
            proj_10 = (1-zz37+z3-z7)/4
            proj_11 = (1+zz37-z3-z7)/4
            
            if label_list[j] == 0:
                cost += proj_00
            elif label_list[j] == 1:
                cost += proj_01
            elif label_list[j] == 2:
                cost += proj_10
            else:
                cost += proj_11

            # adding extra terms to equalize incorrect classes might help in the optimization, 
            # although it is not required
            """
            if label_list[j] == 0:
                cost += proj_00 + ((proj_01-proj_10)**2 + (proj_01-proj_11)**2 + (proj_10-proj_11)**2)/3
            elif label_list[j] == 1:
                cost += proj_01 + ((proj_00-proj_10)**2 + (proj_00-proj_11)**2 + (proj_10-proj_11)**2)/3
            elif label_list[j] == 2:
                cost += proj_10 + ((proj_00-proj_01)**2 + (proj_00-proj_11)**2 + (proj_01-proj_11)**2)/3
            else:
                cost += proj_11 + ((proj_00-proj_01)**2 + (proj_00-proj_10)**2 + (proj_01-proj_10)**2)/3
            """

        return cost/training_data
    
    def accuracy(params):
        accuracy_data = 0
        circuit = models.Circuit(nqubits)
        
        convolutional_layer(circuit, 0, 1, params[:13])
        convolutional_layer(circuit, 2, 3, params[:13])
        convolutional_layer(circuit, 4, 5, params[:13])
        convolutional_layer(circuit, 6, 7, params[:13])
        
        convolutional_layer(circuit, 1, 2, params[:13])
        convolutional_layer(circuit, 3, 4, params[:13])
        convolutional_layer(circuit, 5, 6, params[:13])
        convolutional_layer(circuit, 7, 0, params[:13])
        
        pooling_layer(circuit, 0, 1, params[13:16])
        pooling_layer(circuit, 2, 3, params[13:16])
        pooling_layer(circuit, 4, 5, params[13:16])
        pooling_layer(circuit, 6, 7, params[13:16])
        
        convolutional_layer(circuit, 1, 3, params[16:29])
        convolutional_layer(circuit, 5, 7, params[16:29])
        
        convolutional_layer(circuit, 3, 5, params[16:29])
        convolutional_layer(circuit, 7, 1, params[16:29])
        
        pooling_layer(circuit, 1, 3, params[29:32])
        pooling_layer(circuit, 5, 7, params[29:32])
        
        convolutional_layer(circuit, 3, 7, params[32:45])
        
        for j in range(training_data):
            final_state = circuit(gs_list[j]).state()

            z3 = np.real(ham3.expectation(final_state))
            z7 = np.real(ham7.expectation(final_state))
            zz37 = np.real(ham37.expectation(final_state))
            
            proj_00 = (1+zz37+z3+z7)/4
            proj_01 = (1-zz37-z3+z7)/4
            proj_10 = (1-zz37+z3-z7)/4
            proj_11 = (1+zz37-z3-z7)/4
            
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
                

        return accuracy_data*100/training_data
        
    
    nparams = 45
    nqubits = 8
    acc = 0
    
    label_list = np.loadtxt(f"{training_data}_training_data/LABELS_{training_data}")
    gs_list = np.load(f"{training_data}_training_data/train_groundstates.npy", allow_pickle=True)  
    
    ham7 = MPO_7()
    ham3 = MPO_3()
    ham37 = MPO_37()
        
    while acc < accuracy_training:
        initial_params = np.random.uniform(0, 2 * np.pi, nparams)
        xopt = cma.fmin2(loss, initial_params, 0.7, options={'tolfun': 1e-4})
        print(xopt[1].result.fbest)
        print(xopt[1].result.xbest)
        
        np.savetxt(f"BEST_PARAMS_j1j2_{training_data}", [xopt[1].result.xbest], newline='')
    
        acc = accuracy(xopt[1].result.xbest)
        print(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default=5, type=int)
    parser.add_argument("--accuracy_training", default=100, type=float)
    args = parser.parse_args()
    main(**vars(args))
