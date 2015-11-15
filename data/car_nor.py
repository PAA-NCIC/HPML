l = []
for x in range(7):
    l.append(set())
with open(r'car.data') as somefile:
    for line in somefile:
        L = line.split(',')
        for i, item in enumerate(L):
            l[i].add(item)
for x in range(7):
    l[x] = list(l[x])
    print l[x];

f = open('car_nor.csv','w')
with open(r'car.data') as somefile:
    for line in somefile:
        L = line.split(',')
        for i, item in enumerate(L):
            for j, candi in enumerate(l[i]):
                if(item == candi):
                    f.write(str(j))
            if(i == 6):
                f.write('\n');
            else:
                f.write(',');


