import matplotlib
matplotlib.use('Agg')
import sys
import os
import matplotlib.pyplot as plt

def plot_log_file(path):
    f = open(path)
    # loss_dict = {}
    num_batch = -1
    eval_epochs = {}
    train_iters = {}
    num_batch_cnt = 0
    for line in f.readlines()[:-1]:
        if 'num_batch' in line:
            num_batch_cnt += 1
            num_batch = int(line.strip().replace(',', '').replace(' ', '').split(':')[-1])
        if '[Train] @ ' in line:
        # if line.startswith('[Train]'):
            line = line.split('[Train] @ ')[1]
            line = line.strip().replace(',', '').replace(':', '').strip()
            attr = line.split()
            for i in range(len(attr)):
                if i % 2 == 0:
                    # if 'loss' in attr[i].lower():
                    #     if attr[i] not in loss_dict:
                    #         loss_dict[attr[i]] = []
                    #     loss_dict[attr[i]].append(float(attr[i+1]))
                    # else:
                    #     if attr not in
                    if attr[i] not in train_iters:
                        train_iters[attr[i]] = []
                    train_iters[attr[i]].append(float(attr[i+1]))
        elif '[Eval] @ ' in line: # line.startswith('[Eval]'):
            line = line.split('[Eval] @ ')[1]
            line = line.strip().replace(',', '').replace(':', '').strip()
            attr = line.split()
            for i in range(len(attr)):
                if i % 2 == 0:
                    if attr[i] not in eval_epochs:
                        eval_epochs[attr[i]] = []
                    eval_epochs[attr[i]].append(float(attr[i + 1]))
        else:
            continue

    if num_batch == -1:
        print 'no num_batch'
        exit(0)
    if num_batch_cnt != 1:
        print 'more than one line containing "num_batch"'
        exit(0)

    if 'iter' not in train_iters:
        return

    colors = ['pink', 'red', 'gold', 'lightyellow', 'palegreen', 'g', 'lightblue', 'blue']
    idx = 0
    k = 5
    x = train_iters['iter']
    for k in train_iters:
        if 'iter' == k.lower():
            continue
        y = train_iters[k]
        plt.plot(x, y, label=k, colors=colors[idx], linewidth=0.5)
        idx += 1
        new_y = []
        new_x = []
        for i in range(len(y)):
            if i < round(k/2.) or len(y) - i < round(k/2.):
                continue
            new_x.append(x[i])
            new_y_tmp = []
            for j in range(-k/2, round(k/2.)):
                new_y_tmp.append(y[i+j])
            new_y.append(sum(new_y_tmp)/k)
        plt.plot(new_x, new_y, label=k, colors=colors[idx], linewidth=1)
        idx += 1



    if 'epoch' in eval_epochs:
        x = [i*num_batch for i in eval_epochs['epoch']]
        for k in eval_epochs:
            if 'epoch' == k.lower() or 'ndcg' in k.lower() or 'map' == k.lower():
                continue
            y = eval_epochs[k]
            plt.plot(x, y, label=k, colors=colors[idx], linewidth=0.5)
            idx += 1
            new_y = []
            new_x = []
            for i in range(len(y)):
                if i < round(k / 2.) or len(y) - i < round(k / 2.):
                    continue
                new_x.append(x[i])
                new_y_tmp = []
                for j in range(-k / 2, round(k / 2.)):
                    new_y_tmp.append(y[i + j])
                new_y.append(sum(new_y_tmp) / k)
            plt.plot(new_x, new_y, colors=colors[idx], label=k, linewidth=1)
            idx += 1

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.savefig(path[:-4].replace('/logs/', '/graph/')+'_loss.pdf')
    plt.close()

    if 'epoch' in eval_epochs:
        # x = [i*num_batch for i in eval_epochs['epoch']]
        for k in eval_epochs:
            if 'ndcg@10' == k.lower():#  or 'map' in k:#  or 'map' in k:
                y = eval_epochs[k]
                plt.plot(x, y, label=k, colors=colors[idx],linewidth=0.5)
                idx += 1
                new_y = []
                new_x = []
                for i in range(len(y)):
                    if i < round(k / 2.) or len(y) - i < round(k / 2.):
                        continue
                    new_x.append(x[i])
                    new_y_tmp = []
                    for j in range(-k / 2, round(k / 2.)):
                        new_y_tmp.append(y[i + j])
                    new_y.append(sum(new_y_tmp) / k)
                plt.plot(new_x, new_y, label=k, colors=colors[idx], linewidth=1)
                idx += 1

        plt.xlabel('Iteration')
        plt.ylabel('Metrics')
        plt.legend(loc=1)
        plt.savefig(path[:-4].replace('/logs/', '/graph/')+'_metrics.pdf')
        plt.close()

    f.close()

if __name__ == '__main__':
    log_path = sys.argv[1]
    if os.path.isfile(log_path) or os.path.isfile(os.path.join(os.path.curdir,log_path)): # file
        print os.path.join(os.path.curdir, log_path)
        try:
            plot_log_file(os.path.join(os.path.curdir,log_path))
        except Exception as e:
            print 'Error:', str(e)
    elif os.path.isdir(log_path) or os.path.isdir(os.path.join(os.path.curdir,log_path)): # dir
        for dirpath, dirnames, filenames in os.walk(log_path):
            for fn in filenames:
                if fn.endswith('.log'):
                    print os.path.join(dirpath, fn)
                    try:
                        plot_log_file(os.path.join(dirpath, fn))
                    except Exception as e:
                        print 'Error:', str(e)
    else:
        print 'path error'


# import sys
# import os
# import matplotlib.pyplot as plt
#
# def plot_log_file(path):
#     f = open(path)
#     # loss_dict = {}
#     num_batch = -1
#     eval_epochs = {}
#     train_iters = {}
#     num_batch_cnt = 0
#     for line in f.readlines()[:-1]:
#         if 'num_batch' in line:
#             num_batch_cnt += 1
#             num_batch = int(line.strip().replace(',', '').replace(' ', '').split(':')[-1])
#         if line.startswith('[Train]'):
#             line = line.strip().replace(',', '').replace(':', '').replace('[Train] @', '')
#             attr = line.split()
#             for i in range(len(attr)):
#                 if i % 2 == 0:
#                     # if 'loss' in attr[i].lower():
#                     #     if attr[i] not in loss_dict:
#                     #         loss_dict[attr[i]] = []
#                     #     loss_dict[attr[i]].append(float(attr[i+1]))
#                     # else:
#                     #     if attr not in
#                     if attr[i] not in train_iters:
#                         train_iters[attr[i]] = []
#                     train_iters[attr[i]].append(float(attr[i+1]))
#         elif line.startswith('[Eval]'):
#             line = line.strip().replace(',', '').replace(':', '').replace('[Eval] @', '')
#             attr = line.split()
#             for i in range(len(attr)):
#                 if i % 2 == 0:
#                     if attr[i] not in eval_epochs:
#                         eval_epochs[attr[i]] = []
#                     eval_epochs[attr[i]].append(float(attr[i + 1]))
#         else:
#             continue
#
#     if num_batch == -1:
#         print 'no num_batch'
#         exit(0)
#     if num_batch_cnt != 1:
#         print 'more than one line containing "num_batch"'
#         exit(0)
#     x = train_iters['iter']
#     for k in train_iters:
#         if 'iter' in k.lower():
#             continue
#         y = train_iters[k]
#         plt.plot(x, y, label=k,linewidth=0.5)
#     #
#     # plt.savefig(path+'_train.pdf')
#     # plt.close()
#
#     x = [i*num_batch for i in eval_epochs['epoch']]
#     for k in eval_epochs:
#         if 'epoch' in k.lower() or 'ndcg' in k.lower() or 'map' in k.lower():
#             continue
#         y = eval_epochs[k]
#         plt.plot(x, y, label=k,linewidth=0.5)
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.legend(loc=1)
#     plt.savefig(path[:-4].replace('/logs/', '/graph/')+'_loss.pdf')
#     plt.close()
#
#     x = [i*num_batch for i in eval_epochs['epoch']]
#     for k in eval_epochs:
#         if 'ndcg@10' in k.lower():#  or 'map' in k:#  or 'map' in k:
#             y = eval_epochs[k]
#             plt.plot(x, y, label=k,linewidth=0.5)
#     plt.xlabel('Iteration')
#     plt.ylabel('Metrics')
#     plt.legend(loc=1)
#     plt.savefig(path[:-4].replace('/logs/', '/graph/')+'_metrics.pdf')
#     plt.close()
#
#     f.close()
#
# if __name__ == '__main__':
#     log_path = sys.argv[1]
#     if os.path.isfile(log_path) or os.path.isfile(os.path.join(os.path.curdir,log_path)): # file
#         # try:
#         plot_log_file(log_path)
#         # except Exception as e:
#         #     print e.message
#     elif os.path.isdir(log_path) or os.path.isdir(os.path.join(os.path.curdir,log_path)): # dir
#         for dirpath, dirnames, filenames in os.walk(log_path):
#             for fn in filenames:
#                 if fn.endswith('.log'):
#                     # try:
#                     plot_log_file(os.path.join(dirpath, fn))
#                     # except Exception as e:
#                     #     print e.message
#     else:
#         print 'path error'
