# library imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# result_csv_path
result_csv = 'results_yolov5.csv'

# read the result_csv and get the epochs, map, loss

# This is a sample for the csv data:
#               epoch,      train/box_loss,      train/obj_loss,      train/cls_loss,   metrics/precision,      metrics/recall,     metrics/mAP_0.5,metrics/mAP_0.5:0.95,        val/box_loss,        val/obj_loss,        val/cls_loss,               x/lr0,               x/lr1,               x/lr2
#                   0,             0.07348,            0.027402,            0.024853,             0.19604,             0.28257,             0.15346,            0.070484,             0.05201,            0.012026,            0.015556,            0.070024,           0.0033307,           0.0033307

df = pd.read_csv(result_csv)

# Strip whitespaces from all the columns
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Print the cleaned DataFrame
#print(df)

# get the epochs, map, loss

epochs = df['epoch'].values
maps = df['metrics/mAP_0.5'].values
losses = df['val/box_loss'].values

# set style
sns.set_style("whitegrid")

# specify font family
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

# set subplot layout
subplots_layout = 'vertical'  # 'vertical' or 'horizontal'

if subplots_layout == 'vertical':
    # create figure with two subplots arranged vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # plot mAP in first subplot
    ax1.plot(epochs, maps, linewidth=2, color='green')
    ax1.set_title('mAP', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Score', fontsize=16)
    ax1.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # plot loss in second subplot
    ax2.plot(epochs, losses, linewidth=2, color='red')
    ax2.set_title('Loss', fontsize=20, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Value', fontsize=16)
    ax2.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)

else:
    # create figure with two subplots arranged horizontally
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # plot mAP in first subplot
    ax1.plot(epochs, maps, linewidth=2, color='green')
    ax1.set_title('mAP', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Score', fontsize=16)
    ax1.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # plot loss in second subplot
    ax2.plot(epochs, losses, linewidth=2, color='red')
    ax2.set_title('Loss', fontsize=20, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Value', fontsize=16)
    ax2.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)

# save plot

plt.savefig('figs/yolov5_vertical.png', dpi=300, bbox_inches='tight')

# show plot
plt.show()