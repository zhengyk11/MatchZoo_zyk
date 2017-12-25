import matplotlib
matplotlib.use('Agg')
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

metriclist = ['ndcg@1','ndcg@3','ndcg@10','recall@10','precision@10','map']
scorefile = open('score.txt','w')
def plot_log_file(path,metric_name):
    f = open(path)
    # loss_dict = {}
    num_batch = -1
    net_name = ''
    eval_epochs = {}
    train_iters = {}
    num_batch_cnt = 0
    num_net_name = 0
    valid_test_num = 0
    eval_cnt = 0
    for line in f.readlines()[:-1]:
        if 'num_batch' in line:
            num_batch_cnt += 1
            num_batch = int(line.strip().replace(',', '').replace(' ', '').split(':')[-1])
        if 'EVAL' in line and 'phase' in line:
            valid_test_num += 1
        if 'net_name' in line:
            num_net_name += 1
            net_name = line.strip().replace(',', '').replace(' ', '').replace('"', '').split(':')[-1].strip()
        if '[Train] @ ' in line:
            line = line.split('[Train] @ ')[1]
            line = line.strip().replace(',', '').replace(':', '').strip()
            attr = line.split()
            for i in range(len(attr)):
                if i % 2 == 0:
                    if attr[i] not in train_iters:
                        train_iters[attr[i]] = []
                    train_iters[attr[i]].append(float(attr[i+1]))
        elif '[Eval] @ ' in line: # line.startswith('[Eval]'):
            eval_id = eval_cnt % valid_test_num
            if eval_id not in eval_epochs:
                eval_epochs[eval_id] = {}
            line = line.split('[Eval] @ ')[1]
            line = line.strip().replace(',', '').replace(':', '').strip()
            attr = line.split()
            for i in range(len(attr)):
                if i % 2 == 0:
                    if attr[i] not in eval_epochs[eval_id]:
                        eval_epochs[eval_id][attr[i]] = []
                    eval_epochs[eval_id][attr[i]].append(float(attr[i + 1]))
            eval_cnt += 1
        else:
            continue

    if num_batch == -1:
        print 'no num_batch'
        exit(0)
    if num_batch_cnt != 1:
        print 'more than one line containing "num_batch"'
        exit(0)
    if net_name == '':
        print 'no net name'
        exit(0)
    if num_net_name != 1:
        print 'more than one line containing "net_name"'
        exit(0)

    if 'iter' not in train_iters:
        return

    colors = ['pink', 'red', 'palegreen', 'g', 'lightblue', 'blue', 'moccasin', 'orange', 'lightgrey', 'grey']
    idx = 0
    plot_cnt = 0
    x = train_iters['iter']
    for k in train_iters:
        if 'iter' == k.lower():
            continue
        y = train_iters[k]
        # print colors[idx]
        x_y_minlen = min(len(x), len(y))
        plt.plot(x[:x_y_minlen], np.log(y[:x_y_minlen]), color=colors[idx%10], linewidth=0.5)
        idx += 1

        new_y, new_x = fun(x[:x_y_minlen], np.log(y[:x_y_minlen]), 200, 100)
        plt.plot(new_x, new_y, label=k, color=colors[idx%10], linewidth=1)
        plot_cnt += 1
        idx += 1

    for eval_id in eval_epochs:
        if 'epoch' in eval_epochs[eval_id]:
            x = [i * num_batch for i in eval_epochs[eval_id]['epoch']]
            for k in eval_epochs[eval_id]:
                if 'loss' not in k:
                    continue
                # if 'epoch' == k.lower() or 'ndcg' in k.lower() or 'map' == k.lower():
                #     continue
                y = eval_epochs[eval_id][k]
                # print colors[idx]
                x_y_minlen = min(len(x), len(y))
                plt.plot(x[:x_y_minlen], np.log(y[:x_y_minlen]), color=colors[idx%10], linewidth=0.5)
                idx += 1
                new_y, new_x = fun(x[:x_y_minlen], np.log(y[:x_y_minlen]), 20, 10)
                plt.plot(new_x, new_y, label=k+'_'+str(eval_id), color=colors[idx%10], linewidth=1)
                plot_cnt += 1
                idx += 1
    
    scorefile.write('%s\t%s\t%s\t'%(path.split('/')[-1].split('_')[0],metric_name,path.split('_')[-1].split('.')[0]))
    # plt.xlabel('#Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=plot_cnt)
    plt.title(net_name)
    plt.grid(axis= 'y')
    plt.savefig(path[:-4].replace('/logs/', '/graph/')+'_loss.pdf')
    plt.savefig(path[:-4].replace('/logs/', '/graph/') + '_loss.png')
    plt.close()
    plot_cnt = 0
    idx = 0
    for eval_id in eval_epochs:
        if 'epoch' in eval_epochs[eval_id]:
            # x = [i*num_batch for i in eval_epochs['epoch']]
            for k in eval_epochs[eval_id]:
                if metric_name == k.lower():  # or 'map' in k:#  or 'map' in k:
                    y = eval_epochs[eval_id][k]
                    # print colors[idx]
                    x_y_minlen = min(len(x), len(y))
                    plt.plot(x[:x_y_minlen], y[:x_y_minlen], color=colors[idx%10], linewidth=0.5)
                    #print max(y[-20:])
                    #scorefile.write('%s\t'%(max(y[-20:])))
                    idx += 1
                    new_y, new_x = fun(x[:x_y_minlen], y[:x_y_minlen], 20, 10)
                    plt.plot(new_x, new_y, label=k+'_'+str(eval_id), color=colors[idx%10], linewidth=1)
                    print new_y[-1]
                    scorefile.write('%s\t'%(new_y[-1]))
                    idx += 1
                    plot_cnt += 1
    scorefile.write('\n')
    # plt.xlabel('#Iteration')
    plt.ylabel(metric_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=plot_cnt)
    plt.title(net_name)
    plt.grid(axis='y')
    plt.savefig(path[:-4].replace('/logs/', '/graph/') + '_metrics_'+metric_name+'.pdf')
    plt.savefig(path[:-4].replace('/logs/', '/graph/') + '_metrics_'+metric_name+'.png')
    plt.close()

    f.close()

def fun(x, y, r, s):
    new_y = []
    new_x = []
    for i in range(0,len(y),s):
        if i < r/2 or len(y) - i < r/2 + 1:
            continue
        new_x.append(x[i])
        new_y_tmp = []
        for j in range(-r/2, r/2 + 1):
            new_y_tmp.append(y[i + j])
        new_y.append(sum(new_y_tmp) / len(new_y_tmp))
    return new_y, new_x

if __name__ == '__main__':
    log_path = sys.argv[1]
    for metric in metriclist:
        #scorefile.write('%s\n' % (metric))
        if os.path.isfile(log_path) or os.path.isfile(os.path.join(os.path.curdir,log_path)): # file
            print os.path.join(os.path.curdir, log_path)
            plot_log_file(os.path.join(os.path.curdir,log_path),metric)
        elif os.path.isdir(log_path) or os.path.isdir(os.path.join(os.path.curdir,log_path)): # dir
            for dirpath, dirnames, filenames in os.walk(log_path):
                for fn in filenames:
                    if fn.endswith('.log'):
                        print os.path.join(dirpath, fn)
                        plot_log_file(os.path.join(dirpath, fn),metric)
    else:
        print 'path error'
