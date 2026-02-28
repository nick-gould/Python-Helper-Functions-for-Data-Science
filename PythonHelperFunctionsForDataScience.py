#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:19:21 2026

@author: nickgould
"""

import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta
import time

import io
import webbrowser

from scipy.stats import norm

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic


# from df2gspread import df2gspread as d2g

retryDuration = 5
numRetries = 5

SCOPES = ['https://www.googleapis.com/auth/drive']

#     myColors = ['#FF0000', '#00FF00', '#0000FF','#FF00FF', '#0F0F0F', '#8F8F8F', '#EFEFEF', '#FF8F8F', '#FF8FFF', '#8FFFFF', '#8F8FFF']



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ThisTime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def SaveMyDataFrame(df, name, location='', includeTimeStamp=True, compress=False):
    if includeTimeStamp:
        timestamp = '_' + ThisTime()
    else:
        timestamp = ''
    if compress:
        extension = '.csv.gz'
    else:
        extension = '.csv'
    df.to_csv(location + name + timestamp + extension, index=False)
    print(f'File Saved {location}{name}{timestamp}{extension}\n')

class timeMyCode:
    def __init__(self):
        self.start = datetime.now()
        self.step = self.start
        self.stepNum = 0
        
    def elapsed(self):
        now = datetime.now()
        print(f'Step {self.stepNum} Elapsed:\t {str(now - self.step)}')
        print(f'Total Elapsed:\t {str(now - self.start)}')
        self.step = now
        self.stepNum = self.stepNum + 1
    
def convertObjectsToCategories(df):
    myObjects = list(df.select_dtypes(include='object').columns)
    df[myObjects] = df[myObjects].astype('category')
    return df

def clearEmptyColumns(df):
    df = df.replace('', np.nan)
    df = df.dropna(axis=1, how='all')
    return df

def openWebpage(url):
    webbrowser.open(url)
    
def getDataAfterDate(df, datecol, start):
    return df.loc[df[datecol] >= start]

def getPeriodStart(day=0, week=-2):
    return (datetime.today() + timedelta(days=-datetime.today().weekday()+day, weeks=week)).strftime("%Y-%m-%d")


def SolveCriticalCurrentCubic(a, b, c, d, root=2):
    # a = df['a'].iloc[0]
    # b = df['b'].iloc[0]
    # c = df['c'].iloc[0]
    # d = df['d'].iloc[0]

    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * (b ** 3) - 9 * a * b * c + 27 * (a ** 2) * d) / (27 * a ** 3)
    discriminant = 18 * a * b * c * d - 4 * (b ** 3) * d + (b ** 2) * (c ** 2) - 4 * a * (c ** 3) - 27 * (a ** 2) * (d ** 2)

    if discriminant > 0: #there are 3 real roots
        return 2 * ((-p / 3) ** (1 / 2)) * np.cos((1 / 3) * np.arccos(((3 * q) / (2 * p)) * ((-3 / p) ** (1 / 2))) - 2 * np.pi * root / 3) - b / (3 * a)
    elif p > 0:
        return -2 * ((p / 3) ** (1 / 2)) * np.sinh((1 / 3) * np.arcsinh(((3 * q) / (2 * p)) * ((3 / p) ** (1 / 2)))) - b / (3 * a)
    else:
        return -2 * (np.abs(q) / q) * ((-p / 3) ** (1 / 2)) * np.cosh((1 / 3) * np.arccosh(((-3 * np.abs(q)) / (2 * p)) * ((-3 / p) ** (1 / 2)))) - b / (3 * a)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
def getAveragesAndSTD(df, mergeintoframe, columns, grouping, merge=True):
    myAverages = np.round(df[grouping + columns].groupby(grouping, group_keys=False).mean(), 1)
    myAverages = myAverages.reset_index()
    myAverages.columns = grouping + [col + ' Mean' for col in columns]

    mySTD = np.round(df[grouping + columns].groupby(grouping, group_keys=False).std(), 1)
    mySTD = mySTD.reset_index()
    mySTD.columns = grouping + [col + ' Standard Deviation' for col in columns]
    
    myMedian = np.round(df[grouping + columns].groupby(grouping, group_keys=False).median(), 1)
    myMedian = myMedian.reset_index()
    myMedian.columns = grouping + [col + ' Median' for col in columns]
    
    myIQR = np.round(df[grouping + columns].groupby(grouping, group_keys=False).quantile(.75) - df[grouping + columns].groupby(grouping, group_keys=False).quantile(.25), 1)
    myIQR = myIQR.reset_index()
    myIQR.columns = grouping + [col + ' Interquartile Range' for col in columns]
    
    if merge:
        return mergeintoframe.merge(myAverages, how='left').merge(mySTD, how='left').merge(myMedian, how='left').merge(myIQR, how='left')
    else:
        return myAverages.merge(mySTD, how='left').merge(myMedian, how='left').merge(myIQR, how='left')

def getMetricsTable(df, metric, groupby='Experiment', alpha = 0.05, rounding=1):
    dfMetricValues = df[[groupby, metric]]
    
    CI = str(int(np.round(100*(1-alpha)))) + '%'
    dfMetricTable = pd.DataFrame()
    dfMetricTable[groupby] = df[groupby].drop_duplicates().reset_index(drop=True)
    dfMetricTable['Count'] = dfMetricValues.groupby([groupby]).count().reset_index()[metric]
    dfMetricTable['Max Value'] = dfMetricValues.groupby([groupby]).max().reset_index()[metric]
    dfMetricTable['Mean'] = dfMetricValues.groupby([groupby]).mean().reset_index()[metric]
    dfMetricTable['Standard Deviation'] = dfMetricValues.groupby([groupby]).std().reset_index()[metric]
    dfMetricTable['Standard Error'] = dfMetricTable['Standard Deviation']/np.sqrt(dfMetricTable['Count'])
    dfMetricTable['Lower ' + CI] = dfMetricTable['Mean'] - norm.ppf(1-alpha/2)*dfMetricTable['Standard Error']
    dfMetricTable['Upper ' + CI] = dfMetricTable['Mean'] + norm.ppf(1-alpha/2)*dfMetricTable['Standard Error']
    
    for col in ['Max Value', 'Mean', 'Standard Deviation', 'Standard Error', 'Lower ' + CI, 'Upper ' + CI]:
        dfMetricTable[col] = np.round(dfMetricTable[col], rounding)
    
    return dfMetricTable

def getPercentileTable(df, metric, groupby='Experiment', rounding=1):
    dfMetricValues = df[[groupby, metric]]
    
    dfMetricTable = pd.DataFrame()
    dfMetricTable[groupby] = df[groupby].drop_duplicates().reset_index(drop=True)
    dfMetricTable['Count'] = dfMetricValues.groupby([groupby]).count().reset_index()[metric]
    dfMetricTable['Max Value'] = dfMetricValues.groupby([groupby]).max().reset_index()[metric]
    dfMetricTable['5th Percentile'] = dfMetricValues.groupby([groupby]).quantile(0.05).reset_index()[metric]
    
    dfMetricTable['25th Percentile'] = dfMetricValues.groupby([groupby]).quantile(0.25).reset_index()[metric]
    dfMetricTable['50th Percentile'] = dfMetricValues.groupby([groupby]).quantile(0.50).reset_index()[metric]
    dfMetricTable['75th Percentile'] = dfMetricValues.groupby([groupby]).quantile(0.75).reset_index()[metric]
    dfMetricTable['95th Percentile'] = dfMetricValues.groupby([groupby]).quantile(0.95).reset_index()[metric]
    dfMetricTable['Interquartile Range'] = dfMetricTable['75th Percentile'] - dfMetricTable['25th Percentile']
    
    for col in [x for x in dfMetricTable.columns if x not in [groupby, 'Count']]:
        dfMetricTable[col] = np.round(dfMetricTable[col], rounding)
    
    return dfMetricTable

def getContingencyTable(df, by, values, title='Contingency Table'):
    myCounts = df[by].value_counts().to_frame().reset_index()
    myCounts.columns = [by, 'Total Count']
    myCounts = myCounts.sort_values(by=[by]).reset_index(drop=True)
    
    myBucketCounts = pd.crosstab(df[by], df[values]).reset_index()
    myBucketCounts[''] = 'Count'
    myPercents = pd.crosstab(df[by], df[values], normalize='index').reset_index()
    myPercents[''] = 'Percents'
    myCounts = myCounts.merge(pd.concat([myBucketCounts, myPercents]))
    myCounts = myCounts.set_index([by, ''])
    
    myCounts = myCounts.style.set_table_attributes("style='display:inline'").set_caption(title).set_table_styles([dict(selector="th",props=[('max-width', '64')])])
    
    return myCounts


def getMosaicPlot(df, by, values, title='Mosaic Plot'):#, figsize=(12,12)):
    df = df.copy(deep=True)
    df[values] = df[values].astype('str')
    figsize = (12, 0.75*len(df[by].unique()))
    myColors = sns.color_palette('colorblind')
    colDict = {}
    for count, value in enumerate(df[values].unique()):
        colDict.update({value: myColors[count]})
        
    props = {}
    for x in df[by].unique():
        for y, col in colDict.items():
            props[(x, y)] = {'color': col}
        
#     return props
    fig, ax = plt.subplots(figsize=figsize)
    mosaic(df, [by, values], horizontal=False, properties=props, gap=.005, ax=ax, label_rotation=0, title=title, labelizer=lambda k: '')
    legenditems = [(plt.Rectangle((0,0),1,1, color=colDict[c]), "%s" %c)
                 for i, c in enumerate(df[values].unique().tolist())]

    plt.legend(*zip(*legenditems), bbox_to_anchor=(1,1), loc="upper left")
#     plt.savefig(f'{values}_mosaic.png')
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getCreds():
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
    return service

def getDriveCreds():
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('drive','v3',credentials=creds, cache_discovery=False)
    return service

def getGoogleSheetData(SheetID, SheetName, verbose=False):
    i = 0
    while i < numRetries:
        service = getCreds()
        i += 1
        try:
            sheet = service.spreadsheets()
            result = sheet.values().get(spreadsheetId=SheetID, range=SheetName).execute()
            values = result.get('values', [])
            i = numRetries
            if not values:
                print(f'No data found for: {SheetName}')
            else:
                if verbose:
                    print(f'Data found for: {SheetName}')
                else:
                    pass
                # return values
                columns = values[0]
                data = list(map(lambda x: dict(enumerate(x)), values[1:]))
                data = [[item.get(i, np.nan) for i in range(len(columns))] for item in data]
                df = pd.DataFrame(data, columns=columns)
                df = df.apply(pd.to_numeric, errors='ignore')
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                return df
        except Exception as e:
            # raise
            print(e)
            print(f'Attempt {i}/{numRetries} Failed :\'(')
            if i < numRetries:
                print(f'Waiting {2**(i-1)*retryDuration} seconds, then retrying')
                time.sleep(2**(i-1)*retryDuration)
            else:
                print('Too many fails, giving up')
            print('\n- - - - - - - - -\n')
            
def uploadGoogleDriveFile(file, fileID):
    service = getDriveCreds()
    try:
        media_content = MediaFileUpload(file, chunksize=1024*1024*16, resumable=False)
    
        service.files().update(
            fileId=fileID,
            media_body=media_content,
            supportsAllDrives=True
            ).execute()
        print(f'{file} uploaded to {fileID} at {ThisTime()}')
        return True
    except Exception as e:
        print(e)
        return False
    
def downloadGoogleDriveFile(newFileName, fileID):
    service = getDriveCreds()
    try:
        request = service.files().get_media(fileId=fileID)
        fh = io.FileIO(newFileName, 'wb')
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024*16)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        # downloader.next_chunk()
        print(f'{newFileName} downloaded from {fileID} at {ThisTime()}')
        return True
    except Exception as e:
        print(e)
        return False
    

def updateDriveCSV(df, fileName, fileID, extension='', compress=False, description=''):
    print(f'{description}\t update started at {ThisTime()}')
    SaveMyDataFrame(df, fileName, includeTimeStamp=False, compress=compress)
    for n in range(numRetries):
        try:
            uploadGoogleDriveFile(fileName + extension, fileID)
            print(f'{description}\t update finished at {ThisTime()}\n')
            break
        except Exception as e:
            print(e)
            print(f'Waiting {retryDuration} [Attempt {n+1}/{numRetries}]')
            time.sleep(retryDuration)
            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            








