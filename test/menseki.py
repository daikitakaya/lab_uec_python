filename = 'output.txt'
f = open(filename, mode = 'w', encoding = 'utf-8')
for i in range(10):
  f.write(str(i) + '\n')
f.close()  