"""Regenerate journal figures from the verified exported results. No training is run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

BASE=Path(__file__).resolve().parent
DATA=BASE/'audit'; OUT=BASE.parent/'Figures';OUT.mkdir(parents=True,exist_ok=True)
STUDIES=['HAR_subject','HAR_Dirichlet','PAMAP2_subject']
NAMES={'HAR_subject':'HAR subjects','HAR_Dirichlet':'HAR Dirichlet','PAMAP2_subject':'PAMAP2 subjects'}
METHODS=['naive','mean_length','client_ema','conditional','estimated_conditional','fixed_k']
LABELS={'naive':'Uncorrected','mean_length':'Mean length','client_ema':'Client EMA','conditional':'Conditional oracle','estimated_conditional':'Probe estimate','fixed_k':'Fixed size'}
COLORS=dict(zip(METHODS,['#9E2929','#91710D','#236744','#246295','#8E4081','#202020']))
MARKERS=dict(zip(METHODS,['x','s','^','o','D','v']))
STYLES=dict(zip(METHODS,['--',':','-.','-',(0,(5,1,1,1)),'-']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,
                    'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
                    'axes.linewidth':.65,'lines.linewidth':1.15,'savefig.dpi':600,'ps.fonttype':42,
                    'pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
test=pd.read_csv(DATA/'verified_final_test.csv');diag=pd.read_csv(DATA/'verified_diagnostics.csv')
history=pd.read_csv(DATA/'verified_history.csv');pairs=pd.read_csv(DATA/'test_paired_recomputed.csv')
matched=pd.read_csv(DATA/'matched_paired_recomputed.csv');subjects=pd.read_csv(DATA/'verified_per_subject.csv')

def save(fig,stem):
    fig.savefig(OUT/f'{stem}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{stem}.eps',dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def letter(ax,s):ax.text(-.10,1.045,s,transform=ax.transAxes,fontweight='bold',fontsize=10)

def method_legend(fig,methods=METHODS,ncol=3,y=-.025):
    handles=[Line2D([0],[0],color=COLORS[m],linestyle=STYLES[m],marker=MARKERS[m],markersize=4,label=LABELS[m]) for m in methods]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False,columnspacing=1.3,handlelength=2.5)

# Fig 1: all 27 finite-payload conditions for each of two scalar corrections.
fig,axes=plt.subplots(1,2,figsize=(6.85,4.8),layout='constrained')
rowlabels=[f'{NAMES[s]}  {p:.2f}' for s in STUDIES for p in [.35,.65,.9]]
for ax,method,part in zip(axes,['mean_length','marginal_oracle'],['a','b']):
    vals=np.array([[diag[(diag.study==s)&(diag.target==p)&(diag.k==k)&(diag.method==method)].signed_projected_bias.item()*100 for k in [32,128,512]] for s in STUDIES for p in [.35,.65,.9]])
    im=ax.imshow(vals,cmap='RdBu',norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=6),aspect='auto',interpolation='nearest')
    ax.set_xticks(range(3),['32','128','512']);ax.set_xlabel('Expected coordinates k')
    ax.set_yticks(range(9),rowlabels if method=='mean_length' else [])
    for r in range(9):
        for c in range(3):ax.text(c,r,f'{vals[r,c]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(vals[r,c])>3.6 else '#111111')
    for y in [2.5,5.5]:ax.axhline(y,color='white',linewidth=2)
    ax.tick_params(length=0);letter(ax,part)
cb=fig.colorbar(im,ax=axes,shrink=.82,pad=.02);cb.set_label('Signed projected bias (%)')
save(fig,'Fig1')

# Fig 2: seed-paired endpoint contrasts, all methods against fixed size.
fig,axes=plt.subplots(1,3,figsize=(6.85,3.35),sharey=False)
for i,(ax,s) in enumerate(zip(axes,STUDIES)):
    for j,m in enumerate(METHODS[:-1]):
        d=pairs[(pairs.study==s)&(pairs.family=='main')&(pairs.method==m)].sort_values('k')
        y=d.mean_difference.to_numpy()*100;x=np.arange(3)+(j-2)*.12
        err=np.array([d.mean_difference-d.ci95_low,d.ci95_high-d.mean_difference])*100
        ax.errorbar(x,y,yerr=err,fmt=MARKERS[m],color=COLORS[m],markersize=4,capsize=2,elinewidth=.85,label=LABELS[m])
    ax.axhline(0,color='#707070',linewidth=.7);ax.set_xticks(range(3),['32','128','512']);ax.set_xlabel('Expected coordinates k')
    ax.grid(axis='y',color='#E0E0E0',linewidth=.5);ax.set_axisbelow(True)
    letter(ax,'abc'[i]);ax.text(.5,1.04,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8.5)
    if i<2:ax.set_ylim(-2.7,.85)
    else:ax.set_ylim(-9,5.5)
axes[0].set_ylabel('Test accuracy difference (pp)')
fig.subplots_adjust(left=.10,right=.99,bottom=.28,top=.89,wspace=.38)
method_legend(fig,METHODS[:-1],3,y=-.015);save(fig,'Fig2')

# Fig 3: full validation trajectories. Means are aligned by training round.
fig,axes=plt.subplots(3,3,figsize=(6.85,6.4),sharey='row')
for i,s in enumerate(STUDIES):
    for j,k in enumerate([32,128,512]):
        ax=axes[i,j]
        for m in METHODS:
            block=history[(history.study==s)&(history.family=='main')&(history.k==k)&(history.method==m)]
            a=block.groupby('round')[['total_wire_bytes','val_accuracy']].mean()
            ax.plot(a.total_wire_bytes/1e6,a.val_accuracy*100,color=COLORS[m],linestyle=STYLES[m],marker=MARKERS[m],markevery=[4,12,24],markersize=3,label=LABELS[m])
        ax.set_ylim(0,100);ax.grid(color='#E7E7E7',linewidth=.5);ax.set_axisbelow(True)
        if i==2:ax.set_xlabel('Total wire MB')
        if j==0:ax.set_ylabel(NAMES[s]+'\nValidation accuracy (%)')
        ax.text(.04,.96,f'{chr(97+i*3+j)}   k = {k}',transform=ax.transAxes,va='top',fontsize=8.5)
fig.subplots_adjust(left=.13,right=.99,bottom=.14,top=.98,wspace=.18,hspace=.20)
method_legend(fig,y=.005);save(fig,'Fig3')

# Fig 4: full attempted-communication accounting at k=128.
components=['gradient_uplink_bytes','downlink_bytes','control_uplink_bytes','probe_uplink_bytes','setup_uplink_bytes']
compnames=['Gradient uplink','Model downlink','State control','Calibration','Normalizer setup']
compcolors=['#3F739D','#B9CCD9','#777777','#C995B5','#E0C774'];hatches=['','///','..','xx','--']
fig,axes=plt.subplots(1,3,figsize=(6.85,3.15))
for i,(ax,s) in enumerate(zip(axes,STUDIES)):
    a=test[(test.study==s)&(test.family=='main')&(test.k==128)].groupby('method').mean(numeric_only=True).reindex(METHODS)
    a['setup_uplink_bytes']=a.uplink_bytes-a.gradient_uplink_bytes-a.control_uplink_bytes-a.probe_uplink_bytes
    bottom=np.zeros(6)
    for c,l,col,h in zip(components,compnames,compcolors,hatches):
        y=a[c].to_numpy()/1e6;ax.bar(np.arange(6),y,bottom=bottom,width=.72,color=col,edgecolor='#444444',linewidth=.35,hatch=h,label=l);bottom+=y
    ax.set_xticks(range(6),['U','M','E','C','P','F']);ax.set_xlabel('Estimator');ax.set_ylim(0,18 if i<2 else 8)
    letter(ax,'abc'[i]);ax.text(.5,1.045,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8.5)
    ax.grid(axis='y',linewidth=.5,color='#E7E7E7');ax.set_axisbelow(True)
axes[0].set_ylabel('Attempted logical wire MB')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005))
fig.subplots_adjust(left=.09,right=.99,bottom=.27,top=.87,wspace=.28);save(fig,'Fig4')

# Fig 5: a point is the maximum recorded at validation rounds for one training seed.
fig,axes=plt.subplots(1,3,figsize=(6.85,3.15),sharey=True)
displaymethods=['conditional','estimated_conditional','fixed_k']
maxima=history[(history.family=='main')&(history['round']>0)].groupby(['study','k','method','seed']).max_ipw_this_round.max().reset_index()
for i,(ax,s) in enumerate(zip(axes,STUDIES)):
    for j,m in enumerate(displaymethods):
        for ki,k in enumerate([32,128,512]):
            vals=maxima[(maxima.study==s)&(maxima.method==m)&(maxima.k==k)].sort_values('seed').max_ipw_this_round.to_numpy()
            x=ki+(j-1)*.24+np.linspace(-.065,.065,len(vals))
            ax.scatter(x,vals,s=13,marker=MARKERS[m],facecolor='white' if m=='conditional' else COLORS[m],edgecolor=COLORS[m],linewidth=.7,zorder=3)
    ax.axhline(50,linestyle=':',color='#707070',linewidth=.8);ax.set_yscale('log');ax.set_ylim(1.8,65)
    ax.set_yticks([2,5,10,20,50],['2','5','10','20','50']);ax.set_xticks(range(3),['32','128','512']);ax.set_xlabel('Expected coordinates k')
    ax.grid(axis='y',linewidth=.5,color='#E7E7E7');letter(ax,'abc'[i]);ax.text(.5,1.045,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8.5)
axes[0].set_ylabel('Maximum logged inverse probability')
fig.subplots_adjust(left=.10,right=.99,bottom=.25,top=.87,wspace=.16)
method_legend(fig,displaymethods,3,y=.005);save(fig,'Fig5')

# Supplementary Fig S1: actual training-client label composition.
fig,axes=plt.subplots(1,3,figsize=(6.85,4.8),layout='constrained')
for i,(ax,s) in enumerate(zip(axes,STUDIES)):
    rows=json.loads((DATA/f'{s}_clients.json').read_text());counts=np.array([x['class_counts'] for x in rows]);prop=counts/counts.sum(1,keepdims=True)
    im=ax.imshow(prop,aspect='auto',vmin=0,vmax=.7,cmap='Blues',interpolation='nearest')
    ax.set_yticks(range(len(rows)),[str(x['client']) for x in rows]);ax.set_xticks(range(prop.shape[1]),[str(i+1) for i in range(prop.shape[1])],rotation=90 if i==2 else 0)
    ax.set_xlabel('Class index');ax.set_ylabel('Training client identifier');ax.text(.5,1.055,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8.5);letter(ax,'abc'[i])
fig.colorbar(im,ax=axes,shrink=.7,pad=.015,label='Proportion of client windows');save(fig,'FigS1')

# Supplementary Fig S3: unseen-subject scores, with all 12 seeds contributing to each interval.
fig,axes=plt.subplots(3,1,figsize=(6.85,6.8))
for i,(ax,s) in enumerate(zip(axes,STUDIES)):
    d=subjects[(subjects.study==s)&(subjects.family=='main')&(subjects.k==128)]
    ids=sorted(d.subject.unique())
    for j,m in enumerate(METHODS):
        a=d[d.method==m].groupby('subject').test_accuracy.agg(['mean','std','count']).reindex(ids)
        ax.errorbar(np.arange(len(ids))+(j-2.5)*.115,a['mean']*100,yerr=stats.t.ppf(.975,11)*a['std']/np.sqrt(a['count'])*100,fmt=MARKERS[m],color=COLORS[m],markersize=3,capsize=1.5,elinewidth=.7)
    ax.set_xticks(range(len(ids)),[str(x) for x in ids]);ax.set_xlabel('Test subject identifier');ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,105)
    letter(ax,'abc'[i]);ax.text(.5,1.05,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8.5);ax.grid(axis='y',linewidth=.5,color='#E7E7E7')
fig.subplots_adjust(left=.10,right=.99,bottom=.15,top=.95,hspace=.53);method_legend(fig,y=.0);save(fig,'FigS3')

# Supplementary Fig S4: PAMAP2 row-normalized confusion matrices, all seeds at k=128.
cms=json.loads((DATA/'verified_confusion_matrices.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(6.85,2.9),layout='constrained')
for i,(ax,m) in enumerate(zip(axes,['naive','conditional','fixed_k'])):
    arrays=[np.array(r['confusion']) for r in cms if r['study']=='PAMAP2_subject' and r['family']=='main' and r['k']==128 and r['method']==m]
    mat=np.sum(arrays,axis=0);pct=mat/mat.sum(axis=1,keepdims=True)*100
    im=ax.imshow(pct,cmap='Blues',vmin=0,vmax=100,interpolation='nearest')
    ax.set_xticks(range(12),range(1,13),fontsize=8,rotation=90);ax.set_yticks(range(12),range(1,13),fontsize=8)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class');letter(ax,'abc'[i]);ax.text(.5,1.08,LABELS[m],transform=ax.transAxes,ha='center',fontsize=8)
fig.colorbar(im,ax=axes,shrink=.62,label='Row percentage');save(fig,'FigS4')

# Supplementary Fig S2: complete matched-time contrasts against fixed size.
fig,axes=plt.subplots(1,3,figsize=(6.85,3.35))
for i,(ax,s) in enumerate(zip(axes,STUDIES)):
    for j,m in enumerate(METHODS[:-1]):
        a=matched[(matched.study==s)&(matched.cost_metric=='sim_seconds')&(matched.method==m)].sort_values('k')
        y=a.mean_difference.to_numpy()*100;err=np.array([a.mean_difference-a.ci95_low,a.ci95_high-a.mean_difference])*100
        ax.errorbar(np.arange(3)+(j-2)*.12,y,yerr=err,fmt=MARKERS[m],color=COLORS[m],markersize=4,capsize=2,elinewidth=.85)
    ax.axhline(0,color='#777777',linewidth=.7);ax.set_xticks(range(3),['32','128','512']);ax.set_xlabel('Expected coordinates k')
    ax.set_ylim((-2.3,1.2) if i<2 else (-7.5,1.3));ax.grid(axis='y',color='#E7E7E7',linewidth=.5)
    letter(ax,'abc'[i]);ax.text(.5,1.04,NAMES[s],transform=ax.transAxes,ha='center',fontsize=8.5)
axes[0].set_ylabel('Validation accuracy difference (pp)')
fig.subplots_adjust(left=.10,right=.99,bottom=.28,top=.89,wspace=.38);method_legend(fig,METHODS[:-1],3,y=-.015);save(fig,'FigS2')
print('Created 9 figures in EPS and 600 dpi PNG')
