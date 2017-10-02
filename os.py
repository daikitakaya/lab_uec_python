import os
path = os.getcwd()
fileList = os.listdir(path)
for i, j in enumerate(fileList):
  if j[-4:] == '.txt':
    print(i, j)