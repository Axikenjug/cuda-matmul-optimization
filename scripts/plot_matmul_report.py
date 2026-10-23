"""Recreate the figures in docs/matmul_optimization_report.md.

Run at the repository/package root:
    python scripts/plot_matmul_report.py
Requirements: pandas, matplotlib, numpy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results'
F = ROOT / 'docs' / 'figures'
F.mkdir(parents=True, exist_ok=True)

T4 = pd.read_csv(D/'t4_mapping_summary.csv')
A100 = pd.read_csv(D/'matmul_a100.csv')
FINAL = pd.read_csv(D/'matmul_final_a100.csv')
TILES = pd.read_csv(D/'tile_size_a100.csv')
REG = pd.read_csv(D/'tile_vs_regtile_a100.csv')

DARK = '#1e304b'
BLUE = '#315c94'
TEAL = '#268c83'
ORANGE = '#ce8f36'
GREY = '#6c7888'
COLORS = {'Naive': GREY, 'Tiled-16': BLUE, 'Register-Tiled R=2': TEAL, 'cuBLAS SGEMM': ORANGE}
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':15,
    'axes.labelsize':11, 'axes.edgecolor':'#94a3b8',
    'axes.labelcolor':DARK, 'xtick.color':DARK,'ytick.color':DARK,
    'axes.spines.top':False, 'axes.spines.right':False,
    'figure.facecolor':'white', 'axes.facecolor':'white',
    'grid.color':'#dfe6ed', 'grid.alpha':.65, 'savefig.facecolor':'white'
})

def make(title, xlabel, ylabel, figsize=(10.2,5.7)):
    fig, ax=plt.subplots(figsize=figsize)
    ax.set_title(title, loc='left', color=DARK, weight='bold', pad=22)
    ax.set_xlabel(xlabel, labelpad=9)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    return fig,ax

def save(fig, filename, note):
    fig.text(.075,.015,note,color='#64748b',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.985])
    fig.savefig(F/filename, dpi=215, bbox_inches='tight')
    plt.close(fig)
    print(F/filename)

# Fig 1 - mapping log latency
x=np.arange(len(T4)); w=.34
row=(T4.row_run1_ms+T4.row_run2_ms)/2
col=(T4.col_run1_ms+T4.col_run2_ms)/2
fig, ax=make('T4: Thread Mapping vs. Kernel Latency','Square matrix size (N x N)','Mean latency (ms)')
ax.bar(x-w/2,row,w,label='Row-major mapping (original baseline)',color=BLUE)
ax.bar(x+w/2,col,w,label='Column-first mapping (diagnostic)',color=ORANGE)
ax.set_yscale('log')
ax.set_ylim(.004,11)
ax.set_xticks(x,[str(z) for z in T4['size']])
for xx,ra,co in zip(x,row,col):
    ax.annotate(f'{co/ra:.2f}x slower',(xx,co),xytext=(0,6),textcoords='offset points',ha='center',color=DARK,fontsize=9,weight='bold')
ax.legend(loc='upper left',frameon=False)
save(fig,'01_t4_thread_mapping.png','Source: t4_mapping_summary.csv | mean of two recorded batches; logarithmic Y axis.')

# Fig 2 A100 intermediate tiled experiment (line speedup metric)
fig, ax=make('A100: Shared-Memory Tiling vs Naive','Square matrix size (N x N)','Speedup over Naive (x)')
ax.plot(A100['size'],A100['speedup'],marker='o',markersize=8,linewidth=2.7,color=BLUE,label='Tiled-16 / Naive speedup')
ax.axhline(1,ls='--',lw=1.2,color=GREY)
ax.set_xscale('log',base=2)
ax.set_xticks(A100['size'],[str(int(n)) for n in A100['size']]); ax.set_ylim(.92,1.85)
for sx,sy in zip(A100['size'],A100['speedup']): ax.annotate(f'{sy:.2f}x',(sx,sy),xytext=(0,9),textcoords='offset points',ha='center',color=DARK,fontsize=10)
save(fig,'02_shared_memory_gain.png','Source: matmul_a100.csv | separate A100 run (not mixed with final-run values).')

# Fig 3 Tile size grouped bar normalized to TILE=16
x=np.arange(len(TILES)); width=.26
fig,ax=make('A100: Tile Size Changes Latency','Square matrix size (N x N)','Latency / Tiled-16 latency (x)')
for j, (name,c,key) in enumerate([('TILE=8',GREY,'tile8_ms'),('TILE=16',BLUE,'tile16_ms'),('TILE=32',TEAL,'tile32_ms')]):
    v=TILES[key]/TILES['tile16_ms']
    ax.bar(x+(j-1)*width,v,width,color=c,label=name)
ax.axhline(1,ls='--',lw=1.2,color='#64748b')
ax.set_xticks(x,[str(int(s)) for s in TILES['size']]); ax.set_ylim(0,1.85)
ax.legend(frameon=False,ncol=3,loc='upper right')
save(fig,'03_tile_size_relative.png','Source: tile_size_a100.csv | below 1.0 means faster than TILE=16; lower is better.')

# Fig 4 Register-Tiled vs tiled speedup line
sx=REG['size']; sy=REG['tiled16_ms']/REG['regtiled16_r2_ms']
fig,ax=make('A100: Register Tiling Benefit Depends on Matrix Size','Square matrix size (N x N)','Tiled-16 / Reg-Tiled-16 latency (x)')
ax.axhline(1,ls='--',lw=1.25,color=GREY)
ax.plot(sx,sy,'o-',color=TEAL,markersize=9,linewidth=2.5)
ax.set_xscale('log',base=2)
ax.set_xticks(sx,[str(int(s)) for s in sx]); ax.set_ylim(.77,1.43)
for s,v in zip(sx,sy): ax.annotate(f'{v:.2f}x',(s,v),xytext=(0,10 if v>1 else -19),textcoords='offset points',ha='center',color=DARK,fontsize=10)
save(fig,'04_register_tiling_tradeoff.png','Source: tile_vs_regtile_a100.csv | TILE=16, R=2; above 1.0 means Register-Tiled faster.')

# Fig 5 final latencies grouped bar log-scale
x=np.arange(len(FINAL)); width=.19
fig,ax=make('A100: Final MatMul Latency Comparison','Square matrix size (N x N)','Kernel latency (ms; log scale)',figsize=(11.2,6.4))
cols=[('Naive','naive_ms'),('Tiled-16','tiled_ms'),('Register-Tiled R=2','reg_tiled_ms'),('cuBLAS SGEMM','cublas_ms')]
for j,(name,key) in enumerate(cols):
    ax.bar(x+(j-1.5)*width, FINAL[key], width, color=COLORS[name], label=name)
ax.set_yscale('log'); ax.set_ylim(.005,1.7)
ax.set_xticks(x,[str(int(s)) for s in FINAL['size']])
ax.legend(frameon=False,ncol=2,loc='upper left')
save(fig,'05_final_latency_grouped.png','Source: matmul_final_a100.csv | one same-session batch per variant/shape; log Y axis.')

# Fig 6 Speedup by stage against naive, grouped bars
x=np.arange(len(FINAL)); width=.25
fig,ax=make('A100: Final Speedup over Naive CUDA','Square matrix size (N x N)','Speedup over Naive (x)',figsize=(10.8,5.9))
for j,(name,key,c) in enumerate([('Tiled-16','tiled_ms',BLUE),('Register-Tiled R=2','reg_tiled_ms',TEAL),('cuBLAS SGEMM','cublas_ms',ORANGE)]):
    vals=FINAL['naive_ms']/FINAL[key]
    ax.bar(x+(j-1)*width,vals,width,label=name,color=c)
ax.set_xticks(x,[str(int(s)) for s in FINAL['size']])
ax.set_ylim(0,7.15); ax.axhline(1,ls='--',lw=1.1,color=GREY)
ax.legend(frameon=False,ncol=3,loc='upper left')
save(fig,'06_final_speedup_grouped.png','Source: matmul_final_a100.csv | 1.0x indicates no gain; higher is better.')

# Fig 7 throughput lines vs size
fig,ax=make('A100: Effective FP32 GEMM Throughput','Square matrix size (N x N)','Effective throughput (TFLOP/s)',figsize=(10.5,5.9))
for name,key in [('Naive','naive_gflops'),('Tiled-16','tiled_gflops'),('Register-Tiled R=2','reg_tiled_gflops'),('cuBLAS SGEMM','cublas_gflops')]:
    ax.plot(FINAL['size'],FINAL[key]/1000,marker='o',linewidth=2.35,markersize=6.8,color=COLORS[name],label=name)
ax.set_xscale('log',base=2)
ax.set_xticks(FINAL['size'],[str(int(s)) for s in FINAL['size']]); ax.set_ylim(0,18)
ax.legend(frameon=False,ncol=2,loc='upper left')
save(fig,'07_final_throughput_lines.png','Source: matmul_final_a100.csv | 2MNK FLOPs / kernel time; achieved, not hardware peak.')
