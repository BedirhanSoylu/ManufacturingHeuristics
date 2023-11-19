# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:44:51 2023

@author: Bedirhan
"""

import lcr_IE_applications_source as ie


database = ie.dataPrep('work_element.txt')
workstations = ie.largestCandidateRule(database, ie.TAKT_TIME)
ie.cycleTimeFounder(database, workstations)
ie.efficieny(ie.cycleTimeFounder(database, workstations), database, workstations)
databaseRPW = ie.rankedWeightedData('work_element.txt')
ie.largestCandidateRule(databaseRPW, ie.TAKT_TIME)

