

import os
b=['your path/DF2K/blur_sub300','b']
n=['your path/DF2K/noise_sub300','n']
j=['your path/DF2K/jpeg_sub300','j']
r=['your path/DF2K/rain_sub300','r']
h=['your path/DF2K/haze_sub300','h']
d=['your path/DF2K/dark_sub300','d']
s=['your path/DF2K/sr_sub300','s']

# order_dir=[j,n,s,b,r,d,h]#jnsbrdh

# order_dir1=[n,b,r,j,h,s,d]#nbrjhsd
# order_dir2=[s,n,b,j,d,r,h]#snbjdrh
# order_dir3=[h,n,d,s,b,j,r]#hndsbjr
# order_dir4=[n,h,b,d,s,r,j]#nhbdsrj
# order_dir5=[d,h,n,b,r,j,s]#dhnbrjs

order_dir1=[s,b,n,j,r,h,d]#dhnbrjs
order_dir2=[r,j,b,s,n,h,d]#dhnbrjs

all=[order_dir1,order_dir2]

for order_dir in all:

    length=len(order_dir)

    log_name=''
    dir_list=[]
    for i in range(length):
        log_name+=order_dir[i][1]
        dir_list.append(order_dir[i][0])

        print(log_name)
       
        log_path="your path/MiOIR/meta_info/MiO-meta-DF2K_300_{}.txt".format(log_name)
        with open(log_path,"w") as f:
            for dir in dir_list:
                num=0
                for i in sorted(os.listdir(dir)):
                    f.write(os.path.join(dir,i)+'\r\n')
                    num+=1
                print(num)
