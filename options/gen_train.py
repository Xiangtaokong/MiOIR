import os
import yaml


def check_dir(dir):
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)
s=['/home/notebook/data/personal/S9053103/CL_data/DF2K/sr_sub300','s']
b=['/home/notebook/data/personal/S9053103/CL_data/DF2K/blur_sub300','b']
n=['/home/notebook/data/personal/S9053103/CL_data/DF2K/noise_sub300','n']
j=['/home/notebook/data/personal/S9053103/CL_data/DF2K/jpeg_sub300','j']
r=['/home/notebook/data/personal/S9053103/CL_data/DF2K/rain_sub300','r']
h=['/home/notebook/data/personal/S9053103/CL_data/DF2K/haze_sub300','h']
d=['/home/notebook/data/personal/S9053103/CL_data/DF2K/dark_sub300','d']


#order_dir0=[s,b,n,j,r,h,d]
order_dir0=[r,j,b,s,n,h,d]


all=[order_dir0]
exp_ids=['004']
ids=0
for order in all:
    length=len(order)
    exp_id=exp_ids[ids]
    ids+=1
    for prompt in ['','EP','AP']:#'','EP','AP'
        
        model_type='Uformer'# 'SRResNet','RRDB','SwinIR','Restormer','Uformer','PromptIR'

        if prompt!='':
            name_=''
            order_name=[]# such as ['d','d+h','d+h+r','d+h+r+j','d+h+r+j+n','d+h+r+j+n+s','d+h+r+j+n+s+b']
            for index in range(7):
                if index>0:
                    name_+='+'+order[index][1]
                else:
                    name_+=order[index][1]
                order_name.append(name_)
                

            source_yaml='/home/notebook/data/personal/S9053103/MiOIR/options/train/{}/train_{}_M_{}.yml'.format(model_type,model_type,prompt)
            save_yml_dir='/home/notebook/data/personal/S9053103/MiOIR/options/train/{}'.format(model_type)
            commda_file=os.path.join('/home/notebook/data/personal/S9053103/MiOIR/run_sh','train_{}_S_{}_{}.sh'.format(model_type,prompt,order_name[length-1]))
            with open(commda_file, "w") as f3:
                f3.write("cd /home/notebook/data/personal/S9053103/MiOIR"+'\n')
            
            for data_d in order_name: #['d','d+h','d+h+r','d+h+r+j','d+h+r+j+n','d+h+r+j+n+s','d+h+r+j+n+s+b']

                check_dir(save_yml_dir)
                with open(source_yaml) as f1:
                    data=yaml.safe_load(f1)        
                
                if model_type=='SRResNet':
                    data['name']='MiO_{}_{}_S_{}_{}_25w'.format(exp_id,model_type,prompt,data_d)
                    if data_d==order_name[length-1]:
                        data['name']=data['name']+'_250w'
                        data['train']['scheduler']['periods']=[250000,250000,250000,250000]
                        data['train']['scheduler']['restart_weights']=[1,1,1,1]
                        data['train']['total_iter']=1000000
                    else:
                        data['train']['scheduler']['periods']=[250000]
                        data['train']['scheduler']['restart_weights']=[1]
                        data['train']['total_iter']=250000
                else:
                    data['name']='MiO_{}_{}_S_{}_{}_10w'.format(exp_id,model_type,prompt,data_d)
                    if data_d==order_name[length-1]:
                        data['name']=data['name']+'_100w'
                        data['train']['scheduler']['periods']=[100000,100000,100000,100000]
                        data['train']['scheduler']['restart_weights']=[1,1,1,1]
                        data['train']['total_iter']=400000
                    else:
                        data['train']['scheduler']['periods']=[100000]
                        data['train']['scheduler']['restart_weights']=[1]
                        data['train']['total_iter']=100000
                
                data['datasets']['train']['meta_info']='/home/notebook/data/personal/S9053103/MiOIR/meta_info/MiO-meta-DF2K_300_{}.txt'.format(data_d.replace('+',''))
                if data_d ==order_name[0]:
                    data['path']['pretrain_network_g']=data['path']['resume_state']
                else:
                    data['path']['pretrain_network_g']='/home/notebook/data/personal/S9053103/MiOIR/experiments/{}/models/net_g_latest.pth'.format(last_name)
                last_name=data['name']
                
                print(data['name'])

                file_path = os.path.join(save_yml_dir, data['name'] + '.yml')
                with open(file_path, "w", encoding="utf-8") as f2:
                    yaml.safe_dump(data, f2, sort_keys=False)

                with open(commda_file, "a+") as f3:
                    if model_type=='SwinIR':
                        f3.write("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt {}".format(file_path) + '\n')
                    else:
                        f3.write("CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt {}".format(file_path) + '\n')


        else:
            name_=''
            order_name=[]#['d','d+h','d+h+r','d+h+r+j','d+h+r+j+n','d+h+r+j+n+s','d+h+r+j+n+s+b']
            for index in range(length):
                if index>0:
                    name_+='+'+order[index][1]
                else:
                    name_+=order[index][1]
                order_name.append(name_)
            source_yaml='/home/notebook/data/personal/S9053103/MiOIR/options/train/{}/train_{}_M.yml'.format(model_type,model_type)
            save_yml_dir='/home/notebook/data/personal/S9053103/MiOIR/options/train/{}'.format(model_type)
            commda_file=os.path.join('/home/notebook/data/personal/S9053103/MiOIR/run_sh','train_{}_S_{}.sh'.format(model_type,order_name[length-1]))
            with open(commda_file, "w") as f3:
                f3.write("cd /home/notebook/data/personal/S9053103/MiOIR"+'\n')

            for data_d in order_name:

                check_dir(save_yml_dir)

                with open(source_yaml) as f1:
                    data=yaml.safe_load(f1)
            
                    
                if model_type=='SRResNet':
                    data['name']='MiO_{}_{}_S_{}_25w'.format(exp_id,model_type,data_d)
                    if data_d==order_name[length-1]:
                        data['name']=data['name']+'_250w'
                        data['train']['scheduler']['periods']=[250000,250000,250000,250000]
                        data['train']['scheduler']['restart_weights']=[1,1,1,1]
                        data['train']['total_iter']=1000000
                    else:
                        data['train']['scheduler']['periods']=[250000]
                        data['train']['scheduler']['restart_weights']=[1]
                        data['train']['total_iter']=250000
                else:
                    data['name']='MiO_{}_{}_S_{}_10w'.format(exp_id,model_type,data_d)
                    if data_d==order_name[length-1]:
                        data['name']=data['name']+'_100w'
                        data['train']['scheduler']['periods']=[100000,100000,100000,100000]
                        data['train']['scheduler']['restart_weights']=[1,1,1,1]
                        data['train']['total_iter']=400000
                    else:
                        data['train']['scheduler']['periods']=[100000]
                        data['train']['scheduler']['restart_weights']=[1]
                        data['train']['total_iter']=100000

                
                data['datasets']['train']['meta_info']='/home/notebook/data/personal/S9053103/MiOIR/meta_info/MiO-meta-DF2K_300_{}.txt'.format(data_d.replace('+',''))
                if data_d ==order_name[0]:
                    data['path']['pretrain_network_g']=data['path']['resume_state']
                else:
                    data['path']['pretrain_network_g']='/home/notebook/data/personal/S9053103/MiOIR/experiments/{}/models/net_g_latest.pth'.format(last_name)
                last_name=data['name']

                print(data['name'])

                

                file_path = os.path.join(save_yml_dir, data['name'] + '.yml')
                with open(file_path, "w", encoding="utf-8") as f2:
                    yaml.safe_dump(data, f2, sort_keys=False)

                with open(commda_file, "a+") as f3:
                    if model_type=='SwinIR':
                        f3.write("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4176 basicsr/train.py -opt {}".format(file_path) + '\n')
                    else:
                        f3.write("CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4176 basicsr/train.py -opt {}".format(file_path) + '\n')






