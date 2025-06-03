#%% 
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
sys.path.append('../../python_scripts')

#%% Global figure settings
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': 'Arial'})
plt.rcParams.update({'axes.titlesize': 10})
plt.rcParams.update({'axes.labelsize': 8})
plt.rcParams.update({'xtick.labelsize': 8})
plt.rcParams.update({'ytick.labelsize': 8})
plt.rcParams.update({'legend.fontsize': 8})
plt.rcParams.update({'figure.dpi': 100})
# Set bar chart color
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

#%% Plot functions

# Define a class

class Plot:
    def __init__(self, settings=None):
        self.settings = settings

    # Function: Plot stats of all models

    # Function: Plot stats of all models
    def models_nfold(self, respaths, modelnames, statname = 'RMSE', ylim = None):

        dfstats = []
        for respath in respaths:
            dfstats.append(pd.read_csv(os.path.join(respath,f'{self.settings.name_target}_nfold_summary_stats.csv')))

        fig, ax = plt.subplots(1,1, figsize = (6,2))
        # plot bars for each model side by side
        count = len(modelnames)
        width = 1/(count+1)
        mid = width * (count-1)/2

        # set color cycle
        # ax.set_prop_cycle(color = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:orange'])

        for df, modelname, idx in zip(dfstats, modelnames, range(0, len(modelnames))):
            hatch = '///' if 'GP' in modelname else ''
            # Other hatch options: https://matplotlib.org/3.1.1/gallery/shapes_and_collections/hatch_style_reference.html

            # Remove +inf values
            df.loc[df[statname] > 1e6, statname] = np.nan

            ax.bar(df.nfold - mid + idx*width, df[statname], label = modelname + ' model', 
                   alpha = 0.6, width = width, edgecolor = 'k', linewidth = 1, hatch = hatch)
            # df.plot(y=statname, x = 'nfold', kind = 'bar', title = f'{statname} for {modelname}', ax=ax)
        ax.set_xlim(0.5, self.settings.nfold+0.5)
        if  ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xticks(range(1,  self.settings.nfold+1))
        ax.set_xticklabels(range(1, self.settings.nfold+1))

        ax.set_xlabel('X-validation fold')
        ax.set_title(f'{statname} for all models')
        ax.legend()
        plt.savefig(os.path.join(self.settings.outpath, f'results_{statname}_nfold.png'))
        plt.show()


    # Function: Plot stats of all models
    def models_nfold_st(self, respaths, modelnames, statname = 'RMSE', ylim = None):

        dfstats = []
        for respath in respaths:
            dfstats.append(pd.read_csv(os.path.join(respath,f'{self.settings.name_target}_nfold_summary_stats.csv')))

        
        fig, ax = plt.subplots(1,1, figsize = (6,2))
        # plot bars for each model side by side
        count = len(modelnames)
        width = 1/(count+1)
        mid = width * (count-1)/2

        count2 = self.settings.nfold_t
        width2 =  width/(count2+1)
        mid2 = width2 * (count2+1)/2
        # set color cycle
        # ax.set_prop_cycle(color = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green', 'tab:orange', 'tab:orange'])

        for df, modelname, idx in zip(dfstats, modelnames, range(0, len(modelnames))):
            hatch = '///' if 'GP' in modelname else ''
            # Other hatch options: https://matplotlib.org/3.1.1/gallery/shapes_and_collections/hatch_style_reference.html

            # Remove +inf values
            df.loc[df[statname] > 1e6, statname] = np.nan

            ax.bar(df.nfold_s - mid + idx*width + (-mid2 + df.nfold_t*width2), df[statname], label = modelname + ' model', 
                   alpha = 0.6, width = width2, edgecolor = 'k', linewidth = 1, hatch = hatch)
            # df.plot(y=statname, x = 'nfold_s', kind = 'bar', title = f'{statname} for {modelname}', ax=ax)
        
        ax.set_xlim(0.5, self.settings.nfold_s+0.5)
        if  ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xticks(range(1,  self.settings.nfold_s+1))
        ax.set_xticklabels(range(1, self.settings.nfold_s+1))

        ax.set_xlabel('X-validation fold')
        ax.set_title(f'{statname} for all models')
        ax.legend()
        plt.savefig(os.path.join(self.settings.outpath, f'results_{statname}_nfold_st.png'))
        plt.show()


    #%% 
    # Function: Plot stats of all models
    def models(self, respaths, modelnames, statname = 'RMSE', ylim = None):

        dfstats = []
        for respath in respaths:
            dfstats.append(pd.read_csv(os.path.join(respath,f'{self.settings.name_target}_nfold_summary_stats.csv')))

        fig, ax = plt.subplots(1,1, figsize = (4,2), dpi=100)

        # set color cycle
        # ax.set_prop_cycle(color = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red'])

        df_out = []
        for df, modelname, idx in zip(dfstats, modelnames, range(0, len(modelnames))):
            hatch = '///' if 'GP' in modelname else ''
            df_mean = df.mean()
            df_std = df.std()
            # Remove +inf values
            if df_mean[statname] > 1e6:
                df_mean[statname] = np.nan
                df_std[statname] = np.nan

            df_out.append(pd.DataFrame({modelname + ' mean': df_mean, modelname + ' std':df_std}).drop('nfold'))

            ax.bar(idx, df_mean[statname], yerr = df_std[statname], label = modelname,
                   alpha = 0.6,  edgecolor = 'k', linewidth = 1, hatch = hatch)

        df_out = pd.concat(df_out, axis = 1)
        df_out.to_csv(os.path.join(self.settings.outpath, 'results_stats.csv'))

        # Grid line for y = 0

        ax.set_xticks(range(0,len(modelnames)))
        ax.set_xticklabels(modelnames, ha = 'center')
        ax.set_ylabel(statname)
        ax.set_title(f'{statname} for all models')
        if (ax.get_ylim()[0]) < 0:
            ax.axhline(y=0, color='k', linestyle='--', lw=1)
        if  ylim is not None:
            ax.set_ylim(ylim)
        fig.subplots_adjust(hspace=0.4, wspace=0.2, left=0.15, right=0.95, bottom=0.12, top=0.88)
        # Set background color to transparent
        # fig.patch.set_facecolor('none')
        # ax.patch.set_facecolor('none')
        
        plt.savefig(os.path.join(self.settings.outpath, f'results_{statname}.png'), dpi=300, transparent=True)
        plt.show()
