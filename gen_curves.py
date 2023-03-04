import os
import pandas as pd

run_dir = 'runs/train/exp2'
csv_path = os.path.join(run_dir, 'results.csv')

#csv_path = 'results.csv'

keys = ['epoch', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']

df = pd.read_csv(csv_path, skipinitialspace=True, usecols=keys)

# set the first column as index 

df.set_index('epoch', inplace=True)


#df.plot()

# save plot as png

plot_path = os.path.join(run_dir, 'map_plot.png')

df.plot().get_figure().savefig(plot_path)