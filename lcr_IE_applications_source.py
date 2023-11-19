# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:07:11 2023

@author: Bedirhan
"""

TOTAL_WORK_TIME = 360
PRODUCTION_TARGET = 400
TAKT_TIME = TOTAL_WORK_TIME / PRODUCTION_TARGET

#Plant Layout Algorithms
###################################################

def dataPrep(file):
    elements = {}
    database = open(file,'r')
    s1 = database.readline()
    while s1:
        key = s1.split('/')[0]
        time = float(s1.split('/')[1])
        precedences = s1.split('/')[2][:-1]
        precedences = precedences.split(',')
        elements[key] = (time,precedences)
        s1 = database.readline()
            
    return elements

def fitPrecedence(dictionary, element,assigned):
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
        
            
    
            
def largestCandidateRule(database,TAKT_TIME):
    database = dict(sorted(database.items(), key = lambda item: item[1][0], reverse= True))
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
        
    
    
def efficieny(cycleTime,database,workstations):
    totalWorkTime = 0
    
    for el in database:
        totalWorkTime += database[el][0]
    
    numStation = len(workstations)
    
    groupedWorkLoad = numStation * cycleTime
    efficiency = (totalWorkTime / groupedWorkLoad) * 100
    efficiency = round(efficiency,2)
    
    return efficiency
        
#######################################²
        
def fitAnyPrecedence(event,assigned,database):
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
            
        #elif fitAnyPrecedence(event, assigned, database) and (event not in assigned) and addable and database[event][1] != ['-']
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
    rankedDatabase = dict(sorted(rankedDatabase.items(), key = lambda item: item[1][2], reverse= True))
    return rankedDatabase
        
        
        
        
        
        
        
       