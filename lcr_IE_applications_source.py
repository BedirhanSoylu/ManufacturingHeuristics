# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:07:11 2023

@author: Bedirhan
"""
import numpy as np
import pandas as pd
import sys

TOTAL_WORK_TIME = 360
PRODUCTION_TARGET = 400
TAKT_TIME = TOTAL_WORK_TIME / PRODUCTION_TARGET



# Plant Layout Algorithms
###################################################


def dataPrep(file):
    elements = {}
    database = open(file, 'r')
    s1 = database.readline()
    while s1:
        key = s1.split('/')[0]
        time = float(s1.split('/')[1])
        precedences = s1.split('/')[2][:-1]
        precedences = precedences.split(',')
        elements[key] = (time, precedences)
        s1 = database.readline()

    return elements


def fitPrecedence(dictionary, element, assigned):
    fit_precedence = True
    for el in dictionary[element][1]:
        if el not in assigned:
            fit_precedence = False

    return fit_precedence


def initializeIteration(database, assigned, totalTime):
    suitable = True
    addable = False
    station = ''
    for el in database:
        if (el not in assigned) and fitPrecedence(database, el, assigned) and database[el][0] <= TAKT_TIME - totalTime and suitable:
            station = el
            suitable = False
            addable = True

    return (addable, station)


def largestCandidateRule(database, TAKT_TIME):
    database = dict(
        sorted(database.items(), key=lambda item: item[1][0], reverse=True))
    workstations = []
    assigned = ['-']
    finished = False
    while not finished:
        tot_time = 0
        workstations.append([])
        appending = initializeIteration(database, assigned, tot_time)
        while appending[0] == True:
            workstations[-1].append(appending[1])
            assigned.append(appending[1])
            tot_time += database[appending[1]][0]
            appending = initializeIteration(database, assigned, tot_time)
        if len(assigned) == len(database) + 1:
            finished = True

    return workstations


def cycleTimeFounder(database, workstations):
    maxTime = 0

    for ws in workstations:
        relativeMax = 0

        for event in ws:
            relativeMax += database[event][0]

        if relativeMax > maxTime:
            maxTime = relativeMax

    return maxTime


def efficieny(cycleTime, database, workstations):
    totalWorkTime = 0

    for el in database:
        totalWorkTime += database[el][0]

    numStation = len(workstations)

    groupedWorkLoad = numStation * cycleTime
    efficiency = (totalWorkTime / groupedWorkLoad) * 100
    efficiency = round(efficiency, 2)

    return efficiency

# Ranked Weighted Position Method
# ²


def fitAnyPrecedence(event, assigned, database):
    precedenceExist = False
    for precedence in database[event][1]:
        if precedence in assigned and not precedenceExist:
            precedenceExist = True

    return precedenceExist


def selectionInitializer(database, assigned):
    addable = True
    for event in database:
        if fitAnyPrecedence(event, assigned, database) and (event not in assigned) and addable and database[event][1] != ['-']:
            stationName = event
            wTime = database[event][0]
            addable = False

        # elif fitAnyPrecedence(event, assigned, database) and (event not in assigned) and addable and database[event][1] != ['-']
    if not addable:
        return (not addable, stationName, wTime)

    else:
        return (not addable,)


def rankedWeightedData(file):

    database = dataPrep(file)
    rankedDatabase = {}

    for event in database:
        assigned = []
        assigned.append(event)
        totSum = database[event][0]
        selection = selectionInitializer(database, assigned)

        while selection[0]:

            totSum += selection[2]
            assigned.append(selection[1])

            selection = selectionInitializer(database, assigned)

        print(assigned)

        rankedDatabase[event] = database[event] + (totSum,)
    rankedDatabase = dict(sorted(rankedDatabase.items(),
                          key=lambda item: item[1][2], reverse=True))
    return rankedDatabase

# Material Handling Heuristic
######################################


def isMin(recentMin, queriedMin):
    if recentMin < queriedMin:
        return False

    else:
        return True
def excelReader(W_i_j_file, h_i_j_file):
    W_i_j = pd.read_excel(W_i_j_file)
    h_i_j = pd.read_excel(h_i_j_file)
    return W_i_j, h_i_j


def materailHeuristic(W_i_j, h_i_j, K_i, capUsage = [0,0,0,0], optCost = 0):   
    if W_i_j.shape[0] == 0:
        capUsage = np.ceil(capUsage)
        totCapCost = optCost + np.sum(np.multiply(K_i,capUsage))
        
        print(f'finished, total cost is {totCapCost}')
    
    else:
        # Creating the if all feasible moves performed table
        equipmentTable = pd.DataFrame(columns=['Equipment', 'q_i', 'h_i_j', 'λ_i', 'λ_i * K_i', 'Sum-W_i_j', 'TotalCost', 'CostperMove'])
        recentMin = 100000
        for equipment in range(1, W_i_j.shape[1]):
            sumW = 0
            sumH = 0
            q_i_rel = 0
            feasibleMoves = []
            for move in range(W_i_j.shape[0]):
                if W_i_j.iloc[move, equipment] != 'M':
                    sumW += W_i_j.iloc[move, equipment]
                    sumH += h_i_j.iloc[move, equipment]
                    q_i_rel += 1
                    feasibleMoves.append(W_i_j.iloc[move, 0])
    
            lambda_i = np.ceil(sumH)
            kapitalCost = lambda_i * K_i[equipment-1]
            totCost = kapitalCost + sumW
            
            if q_i_rel != 0:
                costPMove = totCost / q_i_rel
                
                if isMin(recentMin, costPMove):
                    recentMin = costPMove
                    minIndex = equipment
                    q_i = q_i_rel
                
        data = [W_i_j.columns[equipment], q_i, sumH,
                lambda_i,kapitalCost, sumW,
                totCost, costPMove]
        data = pd.DataFrame(data)
        equipmentTable = pd.concat([equipmentTable,data], ignore_index= True )

        # Sorting and deleting according to feasibility and costivity
        feasEqTable_h = h_i_j[['Move', h_i_j.columns[minIndex]]]
        feasEqTable_W = W_i_j[['Move', W_i_j.columns[minIndex]]]
        feasEqTable_h = feasEqTable_h[feasEqTable_h[feasEqTable_h.columns[1]] != '-']
        feasEqTable_W = feasEqTable_W[feasEqTable_W[feasEqTable_W.columns[1]] != 'M']
        feasEqTable_W.drop(feasEqTable_W.columns[0], axis = 1 ,inplace = True)
        feasEqTable = pd.concat([feasEqTable_h,feasEqTable_W], axis = 1)
        feasEqTable.columns = ['Move', 'h_i_j','W_i_j']
        
        feasEqTable.sort_values(by = 'W_i_j',inplace = True)
    
        # Selecting deletable
        deletableMoves = []
        cumulativeCapUsage = 0
        
    
        for h_j in range(feasEqTable.shape[0]):
            if cumulativeCapUsage + feasEqTable['h_i_j'].iloc[h_j] <= 1:
                deletableMoves.append(feasEqTable['Move'].iloc[h_j])
                cumulativeCapUsage += feasEqTable['h_i_j'].iloc[h_j]
                optCost += feasEqTable['W_i_j'].iloc[h_j]
        
        #Deleting added moves
        capUsage[minIndex - 1] += cumulativeCapUsage
        W_i_j = W_i_j[~W_i_j['Move'].isin(deletableMoves)]
        h_i_j = h_i_j[~h_i_j['Move'].isin(deletableMoves)]
        
        capUsageRec = np.ceil(capUsage)
        totCapCost = optCost + np.sum(np.multiply(K_i,capUsageRec))
        
        print(f'Equipment {minIndex} \nMoves {deletableMoves} \nCapacitiy Usage is {cumulativeCapUsage}\nCurrent Cost is {totCapCost}\n')
        
        materailHeuristic(W_i_j, h_i_j, K_i, capUsage, optCost)
        
    

