# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:56:19 2017

@author: tetsuya
"""

#必要なモジュールのimport
import os

#カレントディレクトリの取得
c_path = os.getcwd()
fileList = os.listdir(c_path)

#************************************************************
#カレントディレクトリのすべてのファイルを表示
print("カレントディレクトリの全てのファイルリストを表示します。")

for i,j in enumerate(fileList):
    print(i,j)

NumFiles = len(fileList)

print("ファイル数は全部で" + str(NumFiles) + "個でした。")
#************************************************************

#************************************************************
#txtファイルの抽出    
print("カレントディレクトリのtxtファイルリストを表示します。")

#リスト内包表記でのtxtファイルリストの抽出
txtFiles = [i for i in fileList if ".txt" in i]

for i,j in enumerate(txtFiles):
    print(i,j)

NumFiles = len(txtFiles)
print("ファイル数は全部で" + str(NumFiles) + "個でした。")
#************************************************************


#************************************************************
#xlsxファイルの抽出    
print("カレントディレクトリのxlsxファイルリストを表示します。")

#リスト内包表記でのxlsxファイルリストの抽出
xlsxFiles = [i for i in fileList if ".xlsx" in i]

for i,j in enumerate(xlsxFiles):
    print(i,j)
    
NumFiles = len(xlsxFiles)
print("ファイル数は全部で" + str(NumFiles) + "個でした。")
#************************************************************


#************************************************************
#csvファイルの抽出    
print("カレントディレクトリのcsvファイルリストを表示します。")

#リスト内包表記でのcsvファイルリストの抽出
csvFiles = [i for i in fileList if ".csv" in i]

for i,j in enumerate(txtFiles):
    print(i,j)

NumFiles = len(csvFiles)
print("ファイル数は全部で" + str(NumFiles) + "個でした。")
#************************************************************

print("プログラムを終了します")
