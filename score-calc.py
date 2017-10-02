# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:26:23 2017

@author: tetsuya
"""
#生徒30人分のスコアの平均値と頻度を出力するプログラム


print("**プログラムを開始します**\n")

#30人分の生徒の5段階評価
scores = [3,5,2,4,3,3,4,4,2,4,4,5,3,2,5,1,3,2,3,3,4,3,2,3,4,5,2,3,1,3]

print("スコアは",len(scores),"人分あります")

print("スコアの平均値を計算します")

#平均値の計算
ave = sum(scores) / len(scores)

print("スコアの平均:", ave)

#各スコアの頻度の出力。ここは繰り返しの表現を学ぶともう少し短く書ける
print("5のスコアの人数は:", scores.count(5))
print("4のスコアの人数は:", scores.count(4))
print("3のスコアの人数は:", scores.count(3))
print("2のスコアの人数は:", scores.count(2))
print("1のスコアの人数は:", scores.count(1))

print("")
print("**プログラムを終了します**")