import numpy as np
parse_server_count = 4
device_ids = [2,4]
new_device_ids = []
count = 0
for i in range(parse_server_count):
    for j in range(len(device_ids)):
        if count>=parse_server_count:
            break
        new_device_ids.append(device_ids[j])
        count+=1
parse_server_ports = [] #3300 3301
begin_port = 3300
for n in range(parse_server_count):#0,1
    parse_server_ports.append(begin_port+n)
print(parse_server_ports,new_device_ids)

y=np.array([[0.1,0.8,0.1],
            [0.1,0.8,0.1],
            [0.1,0.8,0.1]])
y = np.argmax(y,axis=1)
t = np.array([1,1,1])
print(y==t)
print(np.sum(np.array([ True,True,True]))) #3

print(6%600)