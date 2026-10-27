"""Minimal four-panel combination update. Run: python -B generate_combined_radars.py
Dependencies: matplotlib, numpy, Pillow, PyMuPDF; Arial and Times New Roman fonts.
The original plotting code is retained, with an option to return Figure objects and a combined export entry point.
Data, tick placement, and fills follow the original script before label boxes and duplicate legends are removed.
Tissue labels and metric descriptions use the original metric colors: magenta for DSC and blue-violet for HD95.
Radar bodies, fonts, tick positions, line widths, curve colors, fills, and local scales follow the original figures.
In group 02 (ablation), Baseline region fills are removed while curves are retained; the shared legend uses line segments.
Outputs are written only to the script directory: three 600 dpi PNGs and three vector PDFs.
"""
from pathlib import Path
import argparse
import ast
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Patch
import numpy as np
from PIL import Image, ImageChops, ImageOps

OUT = Path(__file__).resolve().parent
DEFAULT_SOURCE = OUT.parent / 'redesigned_legend_only_labels_inside_local_axes_2x'


DATASETS = ['GLI', 'MEN', 'MET', 'PED', 'SSA', 'WMH', 'Stroke', 'MS', 'MRBrainS18']

ANATOMY_DATASETS = ['GLI', 'WMH', 'MEN', 'MRBrainS18']

STRUCTURES = [(1, 'GM'), (2, 'BG'), (3, 'WM'), (5, 'Ven'), (6, 'Cer'), (7, 'BS')]

METHODS = {'main': ['Lesion-Only',
          'BrainSeg',
          'Joint-Baseline',
          'TumorSynth',
          'jSTABL',
          'UNA',
          'MedLesSynth-LD',
          'Ours'],
 'ablation': ['Baseline', 'w/o Simplex', 'w/o FDAR', 'w/o MIPB', 'Ours'],
 'wmh_source': ['WMH-Baseline', 'WMH-Ours']}

COLORS = {'Lesion-Only': '#00A6B2',
 'BrainSeg': '#7B3294',
 'Joint-Baseline': '#4D4D4D',
 'TumorSynth': '#B59A00',
 'jSTABL': '#D81B60',
 'UNA': '#0072B2',
 'MedLesSynth-LD': '#E66100',
 'Ours': '#00845A',
 'Baseline': '#4D4D4D',
 'w/o Simplex': '#0072B2',
 'w/o FDAR': '#E66100',
 'w/o MIPB': '#7B3294',
 'WMH-Baseline': '#7B3294',
 'WMH-Ours': '#00845A'}

METRIC_COLORS = {'DSC': ('#DAC9D3', '#712751'), 'HD95': ('#C9C7DF', '#33287F')}

FAMILY_TITLE = {'main': 'Main comparison', 'ablation': 'Ablation', 'wmh_source': 'WMH-trained'}

FAMILY_FILE = {'main': '01_main_results', 'ablation': '02_ablation', 'wmh_source': '03_wmh_supplement'}

JOBS = [('main', 'MRBrainS18'),
 ('main', 'WMH'),
 ('ablation', None),
 ('ablation', 'MRBrainS18'),
 ('ablation', 'WMH'),
 ('wmh_source', 'MRBrainS18')]

ADJUST_AXES = {('main', 'MRBrainS18', 'DSC'): ['BG', 'WM', 'Ven', 'Cer', 'BS'],
 ('main', 'MRBrainS18', 'HD95'): ['GM', 'BG', 'Ven', 'Cer', 'BS'],
 ('main', 'WMH', 'DSC'): ['GM', 'BG', 'WM', 'Ven', 'Cer', 'BS'],
 ('main', 'WMH', 'HD95'): ['GM', 'BG', 'Ven', 'Cer', 'BS'],
 ('ablation', None, 'DSC'): ['MEN', 'MET'],
 ('ablation', 'MRBrainS18', 'HD95'): ['GM', 'BG', 'Ven', 'BS'],
 ('ablation', 'WMH', 'HD95'): ['Ven'],
 ('wmh_source', 'MRBrainS18', 'HD95'): ['BG']}

SCALE_CONFIG = {('main', 'MRBrainS18', 'GM', 'DSC'): (68.0, 72.0, 0),
 ('main', 'MRBrainS18', 'GM', 'HD95'): (14.400000000000006, 8.8, 1),
 ('main', 'MRBrainS18', 'BG', 'DSC'): (57.0, 65.0, 0),
 ('main', 'MRBrainS18', 'BG', 'HD95'): (13.700000000000003, 12.100000000000001, 1),
 ('main', 'MRBrainS18', 'WM', 'DSC'): (73.0, 81.0, 0),
 ('main', 'MRBrainS18', 'WM', 'HD95'): (3.2, 2.4000000000000004, 1),
 ('main', 'MRBrainS18', 'Ven', 'DSC'): (78.0, 86.0, 0),
 ('main', 'MRBrainS18', 'Ven', 'HD95'): (4.9, 3.3000000000000003, 1),
 ('main', 'MRBrainS18', 'Cer', 'DSC'): (85.0, 89.0, 0),
 ('main', 'MRBrainS18', 'Cer', 'HD95'): (8.200000000000003, 5.800000000000001, 1),
 ('main', 'MRBrainS18', 'BS', 'DSC'): (68.0, 76.0, 0),
 ('main', 'MRBrainS18', 'BS', 'HD95'): (8.8, 7.2, 1),
 ('main', 'WMH', 'GM', 'DSC'): (72.0, 80.0, 0),
 ('main', 'WMH', 'GM', 'HD95'): (12.2, 5.800000000000001, 1),
 ('main', 'WMH', 'BG', 'DSC'): (81.0, 89.0, 0),
 ('main', 'WMH', 'BG', 'HD95'): (2.4000000000000004, 1.6, 1),
 ('main', 'WMH', 'WM', 'DSC'): (86.0, 90.0, 0),
 ('main', 'WMH', 'WM', 'HD95'): (2.3, 1.5, 1),
 ('main', 'WMH', 'Ven', 'DSC'): (78.0, 90.0, 0),
 ('main', 'WMH', 'Ven', 'HD95'): (3.3, 1.7000000000000002, 1),
 ('main', 'WMH', 'Cer', 'DSC'): (84.0, 92.0, 0),
 ('main', 'WMH', 'Cer', 'HD95'): (7.8, 4.6000000000000005, 1),
 ('main', 'WMH', 'BS', 'DSC'): (80.0, 88.0, 0),
 ('main', 'WMH', 'BS', 'HD95'): (3.0, 2.2, 1),
 ('ablation', None, 'GLI', 'DSC'): (83.0, 91.0, 0),
 ('ablation', None, 'GLI', 'HD95'): (17.0, 9.0, 0),
 ('ablation', None, 'MEN', 'DSC'): (55.0, 59.0, 0),
 ('ablation', None, 'MEN', 'HD95'): (97.0, 41.0, 0),
 ('ablation', None, 'MET', 'DSC'): (62.0, 66.0, 0),
 ('ablation', None, 'MET', 'HD95'): (71.0, 35.0, 0),
 ('ablation', None, 'PED', 'DSC'): (86.0, 90.0, 0),
 ('ablation', None, 'PED', 'HD95'): (6.9, 6.1000000000000005, 1),
 ('ablation', None, 'SSA', 'DSC'): (86.0, 94.0, 0),
 ('ablation', None, 'SSA', 'HD95'): (34.2, 6.2, 1),
 ('ablation', None, 'WMH', 'DSC'): (28.0, 40.0, 0),
 ('ablation', None, 'WMH', 'HD95'): (137.0, 21.0, 0),
 ('ablation', None, 'Stroke', 'DSC'): (43.0, 51.0, 0),
 ('ablation', None, 'Stroke', 'HD95'): (62.0, 18.0, 0),
 ('ablation', None, 'MS', 'DSC'): (25.0, 41.0, 0),
 ('ablation', None, 'MS', 'HD95'): (106.0, 30.0, 0),
 ('ablation', None, 'MRBrainS18', 'DSC'): (18.0, 26.0, 0),
 ('ablation', None, 'MRBrainS18', 'HD95'): (182.0, 38.0, 0),
 ('ablation', 'MRBrainS18', 'GM', 'DSC'): (68.0, 72.0, 0),
 ('ablation', 'MRBrainS18', 'GM', 'HD95'): (9.700000000000001, 9.3, 1),
 ('ablation', 'MRBrainS18', 'BG', 'DSC'): (61.0, 65.0, 0),
 ('ablation', 'MRBrainS18', 'BG', 'HD95'): (13.000000000000002, 12.200000000000001, 1),
 ('ablation', 'MRBrainS18', 'WM', 'DSC'): (76.0, 80.0, 0),
 ('ablation', 'MRBrainS18', 'WM', 'HD95'): (3.2, 2.4000000000000004, 1),
 ('ablation', 'MRBrainS18', 'Ven', 'DSC'): (82.0, 86.0, 0),
 ('ablation', 'MRBrainS18', 'Ven', 'HD95'): (4.1000000000000005, 3.3000000000000003, 1),
 ('ablation', 'MRBrainS18', 'Cer', 'DSC'): (85.0, 89.0, 0),
 ('ablation', 'MRBrainS18', 'Cer', 'HD95'): (6.4, 6.0, 1),
 ('ablation', 'MRBrainS18', 'BS', 'DSC'): (67.0, 75.0, 0),
 ('ablation', 'MRBrainS18', 'BS', 'HD95'): (8.100000000000001, 7.300000000000001, 1),
 ('ablation', 'WMH', 'GM', 'DSC'): (72.0, 80.0, 0),
 ('ablation', 'WMH', 'GM', 'HD95'): (6.800000000000001, 6.4, 1),
 ('ablation', 'WMH', 'BG', 'DSC'): (81.0, 89.0, 0),
 ('ablation', 'WMH', 'BG', 'HD95'): (2.4000000000000004, 1.6, 1),
 ('ablation', 'WMH', 'WM', 'DSC'): (86.0, 90.0, 0),
 ('ablation', 'WMH', 'WM', 'HD95'): (1.9, 1.5, 1),
 ('ablation', 'WMH', 'Ven', 'DSC'): (85.0, 89.0, 0),
 ('ablation', 'WMH', 'Ven', 'HD95'): (2.2, 1.8, 1),
 ('ablation', 'WMH', 'Cer', 'DSC'): (84.0, 92.0, 0),
 ('ablation', 'WMH', 'Cer', 'HD95'): (5.6000000000000005, 4.800000000000001, 1),
 ('ablation', 'WMH', 'BS', 'DSC'): (80.0, 88.0, 0),
 ('ablation', 'WMH', 'BS', 'HD95'): (3.0, 2.2, 1),
 ('wmh_source', 'MRBrainS18', 'GM', 'DSC'): (76.0, 84.0, 0),
 ('wmh_source', 'MRBrainS18', 'GM', 'HD95'): (3.4, 3.0, 1),
 ('wmh_source', 'MRBrainS18', 'BG', 'DSC'): (60.0, 68.0, 0),
 ('wmh_source', 'MRBrainS18', 'BG', 'HD95'): (12.600000000000001, 11.8, 1),
 ('wmh_source', 'MRBrainS18', 'WM', 'DSC'): (78.0, 82.0, 0),
 ('wmh_source', 'MRBrainS18', 'WM', 'HD95'): (2.3000000000000003, 1.9000000000000001, 1),
 ('wmh_source', 'MRBrainS18', 'Ven', 'DSC'): (79.0, 87.0, 0),
 ('wmh_source', 'MRBrainS18', 'Ven', 'HD95'): (4.9, 4.1000000000000005, 1),
 ('wmh_source', 'MRBrainS18', 'Cer', 'DSC'): (87.0, 91.0, 0),
 ('wmh_source', 'MRBrainS18', 'Cer', 'HD95'): (3.6, 3.2, 1),
 ('wmh_source', 'MRBrainS18', 'BS', 'DSC'): (74.0, 78.0, 0),
 ('wmh_source', 'MRBrainS18', 'BS', 'HD95'): (7.800000000000001, 7.4, 1),
 ('main', None, 'GLI', 'DSC'): (69.0, 93.0, 0),
 ('main', None, 'GLI', 'HD95'): (25.0, 9.0, 0),
 ('main', None, 'MEN', 'DSC'): (31.0, 59.0, 0),
 ('main', None, 'MEN', 'HD95'): (119.0, 39.0, 0),
 ('main', None, 'MET', 'DSC'): (42.0, 66.0, 0),
 ('main', None, 'MET', 'HD95'): (132.0, 28.0, 0),
 ('main', None, 'PED', 'DSC'): (51.0, 91.0, 0),
 ('main', None, 'PED', 'HD95'): (37.0, 5.0, 0),
 ('main', None, 'SSA', 'DSC'): (70.0, 94.0, 0),
 ('main', None, 'SSA', 'HD95'): (42.0, 6.0, 0),
 ('main', None, 'WMH', 'DSC'): (1.0, 41.0, 0),
 ('main', None, 'WMH', 'HD95'): (305.0, 21.0, 0),
 ('main', None, 'Stroke', 'DSC'): (23.0, 51.0, 0),
 ('main', None, 'Stroke', 'HD95'): (129.0, 17.0, 0),
 ('main', None, 'MS', 'DSC'): (1.0, 41.0, 0),
 ('main', None, 'MS', 'HD95'): (257.0, 29.0, 0),
 ('main', None, 'MRBrainS18', 'DSC'): (2.0, 26.0, 0),
 ('main', None, 'MRBrainS18', 'HD95'): (334.0, 38.0, 0),
 ('main', 'GLI', 'GM', 'DSC'): (77.0, 97.0, 0),
 ('main', 'GLI', 'GM', 'HD95'): (2.6, 1.0, 1),
 ('main', 'GLI', 'BG', 'DSC'): (64.0, 96.0, 0),
 ('main', 'GLI', 'BG', 'HD95'): (13.0, 1.0, 0),
 ('main', 'GLI', 'WM', 'DSC'): (80.0, 96.0, 0),
 ('main', 'GLI', 'WM', 'HD95'): (3.9000000000000004, 1.1, 1),
 ('main', 'GLI', 'Ven', 'DSC'): (80.0, 96.0, 0),
 ('main', 'GLI', 'Ven', 'HD95'): (9.0, 1.0, 0),
 ('main', 'GLI', 'Cer', 'DSC'): (91.0, 99.0, 0),
 ('main', 'GLI', 'Cer', 'HD95'): (4.300000000000001, 1.1, 1),
 ('main', 'GLI', 'BS', 'DSC'): (70.0, 98.0, 0),
 ('main', 'GLI', 'BS', 'HD95'): (17.0, 1.0, 0),
 ('main', 'MEN', 'GM', 'DSC'): (80.0, 96.0, 0),
 ('main', 'MEN', 'GM', 'HD95'): (2.6, 1.0, 1),
 ('main', 'MEN', 'BG', 'DSC'): (65.0, 97.0, 0),
 ('main', 'MEN', 'BG', 'HD95'): (13.0, 1.0, 0),
 ('main', 'MEN', 'WM', 'DSC'): (84.0, 96.0, 0),
 ('main', 'MEN', 'WM', 'HD95'): (3.0, 1.0, 1),
 ('main', 'MEN', 'Ven', 'DSC'): (80.0, 96.0, 0),
 ('main', 'MEN', 'Ven', 'HD95'): (9.0, 1.0, 0),
 ('main', 'MEN', 'Cer', 'DSC'): (90.0, 98.0, 0),
 ('main', 'MEN', 'Cer', 'HD95'): (3.2, 1.2000000000000002, 1),
 ('main', 'MEN', 'BS', 'DSC'): (70.0, 98.0, 0),
 ('main', 'MEN', 'BS', 'HD95'): (17.0, 1.0, 0),
 ('ablation', 'GLI', 'GM', 'DSC'): (92.0, 96.0, 0),
 ('ablation', 'GLI', 'GM', 'HD95'): (1.8, 1.0, 1),
 ('ablation', 'GLI', 'BG', 'DSC'): (91.0, 95.0, 0),
 ('ablation', 'GLI', 'BG', 'HD95'): (1.6, 1.2000000000000002, 1),
 ('ablation', 'GLI', 'WM', 'DSC'): (92.0, 96.0, 0),
 ('ablation', 'GLI', 'WM', 'HD95'): (1.5, 1.1, 1),
 ('ablation', 'GLI', 'Ven', 'DSC'): (91.0, 95.0, 0),
 ('ablation', 'GLI', 'Ven', 'HD95'): (2.0, 1.2000000000000002, 1),
 ('ablation', 'GLI', 'Cer', 'DSC'): (91.0, 99.0, 0),
 ('ablation', 'GLI', 'Cer', 'HD95'): (1.6, 1.2000000000000002, 1),
 ('ablation', 'GLI', 'BS', 'DSC'): (93.0, 97.0, 0),
 ('ablation', 'GLI', 'BS', 'HD95'): (1.6, 1.2000000000000002, 1),
 ('ablation', 'MEN', 'GM', 'DSC'): (92.0, 96.0, 0),
 ('ablation', 'MEN', 'GM', 'HD95'): (1.8, 1.0, 1),
 ('ablation', 'MEN', 'BG', 'DSC'): (92.0, 96.0, 0),
 ('ablation', 'MEN', 'BG', 'HD95'): (1.5, 1.1, 1),
 ('ablation', 'MEN', 'WM', 'DSC'): (92.0, 96.0, 0),
 ('ablation', 'MEN', 'WM', 'HD95'): (1.5, 1.1, 1),
 ('ablation', 'MEN', 'Ven', 'DSC'): (92.0, 96.0, 0),
 ('ablation', 'MEN', 'Ven', 'HD95'): (1.9000000000000001, 1.1, 1),
 ('ablation', 'MEN', 'Cer', 'DSC'): (94.0, 98.0, 0),
 ('ablation', 'MEN', 'Cer', 'HD95'): (2.0, 1.2000000000000002, 1),
 ('ablation', 'MEN', 'BS', 'DSC'): (89.0, 97.0, 0),
 ('ablation', 'MEN', 'BS', 'HD95'): (1.5, 1.1, 1),
 ('wmh_source', None, 'GLI', 'DSC'): (67.0, 83.0, 0),
 ('wmh_source', None, 'GLI', 'HD95'): (29.3, 11.3, 1),
 ('wmh_source', None, 'MEN', 'DSC'): (44.0, 60.0, 0),
 ('wmh_source', None, 'MEN', 'HD95'): (54.80000000000001, 26.8, 1),
 ('wmh_source', None, 'MET', 'DSC'): (61.0, 69.0, 0),
 ('wmh_source', None, 'MET', 'HD95'): (55.0, 15.0, 0),
 ('wmh_source', None, 'PED', 'DSC'): (48.0, 76.0, 0),
 ('wmh_source', None, 'PED', 'HD95'): (95.0, 23.0, 0),
 ('wmh_source', None, 'SSA', 'DSC'): (62.0, 86.0, 0),
 ('wmh_source', None, 'SSA', 'HD95'): (31.4, 13.4, 1),
 ('wmh_source', None, 'WMH', 'DSC'): (72.0, 76.0, 0),
 ('wmh_source', None, 'WMH', 'HD95'): (8.000000000000004, 4.0, 0),
 ('wmh_source', None, 'Stroke', 'DSC'): (46.0, 58.0, 0),
 ('wmh_source', None, 'Stroke', 'HD95'): (18.0, 14.0, 0),
 ('wmh_source', None, 'MS', 'DSC'): (5.0, 61.0, 0),
 ('wmh_source', None, 'MS', 'HD95'): (43.69999999999999, 19.700000000000003, 1),
 ('wmh_source', None, 'MRBrainS18', 'DSC'): (50.0, 58.0, 0),
 ('wmh_source', None, 'MRBrainS18', 'HD95'): (23.500000000000004, 18.7, 1),
 ('wmh_source', 'GLI', 'GM', 'DSC'): (73.0, 81.0, 0),
 ('wmh_source', 'GLI', 'GM', 'HD95'): (8.4, 7.6000000000000005, 1),
 ('wmh_source', 'GLI', 'BG', 'DSC'): (70.0, 78.0, 0),
 ('wmh_source', 'GLI', 'BG', 'HD95'): (9.600000000000001, 6.4, 1),
 ('wmh_source', 'GLI', 'WM', 'DSC'): (83.0, 87.0, 0),
 ('wmh_source', 'GLI', 'WM', 'HD95'): (3.9000000000000004, 3.1, 1),
 ('wmh_source', 'GLI', 'Ven', 'DSC'): (76.0, 80.0, 0),
 ('wmh_source', 'GLI', 'Ven', 'HD95'): (14.0, 6.0, 0),
 ('wmh_source', 'GLI', 'Cer', 'DSC'): (75.0, 83.0, 0),
 ('wmh_source', 'GLI', 'Cer', 'HD95'): (20.0, 12.0, 0),
 ('wmh_source', 'GLI', 'BS', 'DSC'): (63.0, 71.0, 0),
 ('wmh_source', 'GLI', 'BS', 'HD95'): (12.8, 12.4, 1),
 ('wmh_source', 'WMH', 'GM', 'DSC'): (91.0, 95.0, 0),
 ('wmh_source', 'WMH', 'GM', 'HD95'): (1.6, 1.2000000000000002, 1),
 ('wmh_source', 'WMH', 'BG', 'DSC'): (92.0, 96.0, 0),
 ('wmh_source', 'WMH', 'BG', 'HD95'): (1.4, 1.0, 1),
 ('wmh_source', 'WMH', 'WM', 'DSC'): (92.0, 96.0, 0),
 ('wmh_source', 'WMH', 'WM', 'HD95'): (1.4, 1.0, 1),
 ('wmh_source', 'WMH', 'Ven', 'DSC'): (89.0, 97.0, 0),
 ('wmh_source', 'WMH', 'Ven', 'HD95'): (1.4, 1.0, 1),
 ('wmh_source', 'WMH', 'Cer', 'DSC'): (94.0, 98.0, 0),
 ('wmh_source', 'WMH', 'Cer', 'HD95'): (1.4, 1.0, 1),
 ('wmh_source', 'WMH', 'BS', 'DSC'): (93.0, 97.0, 0),
 ('wmh_source', 'WMH', 'BS', 'HD95'): (1.4, 1.0, 1),
 ('wmh_source', 'MEN', 'GM', 'DSC'): (78.0, 82.0, 0),
 ('wmh_source', 'MEN', 'GM', 'HD95'): (7.4, 6.6000000000000005, 1),
 ('wmh_source', 'MEN', 'BG', 'DSC'): (72.0, 80.0, 0),
 ('wmh_source', 'MEN', 'BG', 'HD95'): (7.800000000000001, 5.0, 1),
 ('wmh_source', 'MEN', 'WM', 'DSC'): (80.0, 88.0, 0),
 ('wmh_source', 'MEN', 'WM', 'HD95'): (3.4000000000000004, 2.6, 1),
 ('wmh_source', 'MEN', 'Ven', 'DSC'): (73.0, 81.0, 0),
 ('wmh_source', 'MEN', 'Ven', 'HD95'): (8.0, 4.0, 0),
 ('wmh_source', 'MEN', 'Cer', 'DSC'): (82.0, 86.0, 0),
 ('wmh_source', 'MEN', 'Cer', 'HD95'): (18.0, 10.0, 0),
 ('wmh_source', 'MEN', 'BS', 'DSC'): (76.0, 80.0, 0),
 ('wmh_source', 'MEN', 'BS', 'HD95'): (15.0, 7.0, 0)}

DATA = {('main', 'GLI', 'Lesion-Only', 4, 'DSC'): (92.8, 6.9),
 ('main', 'GLI', 'Lesion-Only', 4, 'HD95'): (15.2, 20.73),
 ('main', 'GLI', 'BrainSeg', 4, 'DSC'): (73.1, 18.2),
 ('main', 'GLI', 'BrainSeg', 4, 'HD95'): (25.84, 16.45),
 ('main', 'GLI', 'Joint-Baseline', 1, 'DSC'): (96.3, 1.6),
 ('main', 'GLI', 'Joint-Baseline', 2, 'DSC'): (95.9, 3.1),
 ('main', 'GLI', 'Joint-Baseline', 3, 'DSC'): (96.1, 2.0),
 ('main', 'GLI', 'Joint-Baseline', 4, 'DSC'): (91.2, 9.9),
 ('main', 'GLI', 'Joint-Baseline', 5, 'DSC'): (95.7, 2.7),
 ('main', 'GLI', 'Joint-Baseline', 6, 'DSC'): (98.5, 2.8),
 ('main', 'GLI', 'Joint-Baseline', 7, 'DSC'): (97.6, 3.1),
 ('main', 'GLI', 'Joint-Baseline', 1, 'HD95'): (1.05, 0.23),
 ('main', 'GLI', 'Joint-Baseline', 2, 'HD95'): (1.11, 0.43),
 ('main', 'GLI', 'Joint-Baseline', 3, 'HD95'): (1.09, 0.3),
 ('main', 'GLI', 'Joint-Baseline', 4, 'HD95'): (16.41, 29.57),
 ('main', 'GLI', 'Joint-Baseline', 5, 'HD95'): (1.11, 0.42),
 ('main', 'GLI', 'Joint-Baseline', 6, 'HD95'): (1.19, 1.88),
 ('main', 'GLI', 'Joint-Baseline', 7, 'HD95'): (1.21, 1.7),
 ('main', 'GLI', 'TumorSynth', 1, 'DSC'): (92.7, 2.3),
 ('main', 'GLI', 'TumorSynth', 2, 'DSC'): (93.5, 4.2),
 ('main', 'GLI', 'TumorSynth', 3, 'DSC'): (93.0, 2.4),
 ('main', 'GLI', 'TumorSynth', 4, 'DSC'): (84.3, 14.2),
 ('main', 'GLI', 'TumorSynth', 5, 'DSC'): (92.7, 3.6),
 ('main', 'GLI', 'TumorSynth', 6, 'DSC'): (98.1, 1.7),
 ('main', 'GLI', 'TumorSynth', 7, 'DSC'): (96.5, 2.2),
 ('main', 'GLI', 'TumorSynth', 1, 'HD95'): (1.23, 0.39),
 ('main', 'GLI', 'TumorSynth', 2, 'HD95'): (1.24, 0.71),
 ('main', 'GLI', 'TumorSynth', 3, 'HD95'): (1.45, 0.57),
 ('main', 'GLI', 'TumorSynth', 4, 'HD95'): (19.97, 21.2),
 ('main', 'GLI', 'TumorSynth', 5, 'HD95'): (1.25, 0.57),
 ('main', 'GLI', 'TumorSynth', 6, 'HD95'): (1.16, 0.72),
 ('main', 'GLI', 'TumorSynth', 7, 'HD95'): (1.16, 0.82),
 ('main', 'GLI', 'jSTABL', 1, 'DSC'): (79.3, 4.3),
 ('main', 'GLI', 'jSTABL', 2, 'DSC'): (65.7, 6.6),
 ('main', 'GLI', 'jSTABL', 3, 'DSC'): (83.2, 4.7),
 ('main', 'GLI', 'jSTABL', 4, 'DSC'): (89.0, 11.2),
 ('main', 'GLI', 'jSTABL', 5, 'DSC'): (81.6, 9.3),
 ('main', 'GLI', 'jSTABL', 6, 'DSC'): (91.6, 3.1),
 ('main', 'GLI', 'jSTABL', 7, 'DSC'): (70.9, 3.3),
 ('main', 'GLI', 'jSTABL', 1, 'HD95'): (2.59, 0.59),
 ('main', 'GLI', 'jSTABL', 2, 'HD95'): (12.7, 3.12),
 ('main', 'GLI', 'jSTABL', 3, 'HD95'): (3.85, 1.74),
 ('main', 'GLI', 'jSTABL', 4, 'HD95'): (13.75, 16.17),
 ('main', 'GLI', 'jSTABL', 5, 'HD95'): (6.07, 3.58),
 ('main', 'GLI', 'jSTABL', 6, 'HD95'): (3.9, 2.0),
 ('main', 'GLI', 'jSTABL', 7, 'HD95'): (13.75, 1.26),
 ('main', 'GLI', 'UNA', 1, 'DSC'): (96.0, 2.0),
 ('main', 'GLI', 'UNA', 2, 'DSC'): (95.2, 4.0),
 ('main', 'GLI', 'UNA', 3, 'DSC'): (95.9, 2.3),
 ('main', 'GLI', 'UNA', 4, 'DSC'): (91.0, 9.4),
 ('main', 'GLI', 'UNA', 5, 'DSC'): (94.8, 4.0),
 ('main', 'GLI', 'UNA', 6, 'DSC'): (98.4, 2.8),
 ('main', 'GLI', 'UNA', 7, 'DSC'): (97.2, 3.4),
 ('main', 'GLI', 'UNA', 1, 'HD95'): (1.07, 0.29),
 ('main', 'GLI', 'UNA', 2, 'HD95'): (1.21, 0.64),
 ('main', 'GLI', 'UNA', 3, 'HD95'): (1.11, 0.36),
 ('main', 'GLI', 'UNA', 4, 'HD95'): (12.86, 17.21),
 ('main', 'GLI', 'UNA', 5, 'HD95'): (1.27, 0.93),
 ('main', 'GLI', 'UNA', 6, 'HD95'): (1.21, 1.87),
 ('main', 'GLI', 'UNA', 7, 'HD95'): (1.22, 1.69),
 ('main', 'GLI', 'MedLesSynth-LD', 1, 'DSC'): (96.1, 1.9),
 ('main', 'GLI', 'MedLesSynth-LD', 2, 'DSC'): (95.2, 3.8),
 ('main', 'GLI', 'MedLesSynth-LD', 3, 'DSC'): (95.9, 2.3),
 ('main', 'GLI', 'MedLesSynth-LD', 4, 'DSC'): (91.0, 8.7),
 ('main', 'GLI', 'MedLesSynth-LD', 5, 'DSC'): (94.9, 4.0),
 ('main', 'GLI', 'MedLesSynth-LD', 6, 'DSC'): (98.4, 2.9),
 ('main', 'GLI', 'MedLesSynth-LD', 7, 'DSC'): (97.3, 3.3),
 ('main', 'GLI', 'MedLesSynth-LD', 1, 'HD95'): (1.07, 0.28),
 ('main', 'GLI', 'MedLesSynth-LD', 2, 'HD95'): (1.2, 0.67),
 ('main', 'GLI', 'MedLesSynth-LD', 3, 'HD95'): (1.1, 0.35),
 ('main', 'GLI', 'MedLesSynth-LD', 4, 'HD95'): (11.98, 16.16),
 ('main', 'GLI', 'MedLesSynth-LD', 5, 'HD95'): (1.24, 0.89),
 ('main', 'GLI', 'MedLesSynth-LD', 6, 'HD95'): (1.22, 1.9),
 ('main', 'GLI', 'MedLesSynth-LD', 7, 'HD95'): (1.22, 1.69),
 ('main', 'GLI', 'Ours', 1, 'DSC'): (96.0, 1.9),
 ('main', 'GLI', 'Ours', 2, 'DSC'): (95.1, 4.0),
 ('main', 'GLI', 'Ours', 3, 'DSC'): (95.9, 2.3),
 ('main', 'GLI', 'Ours', 4, 'DSC'): (91.3, 8.5),
 ('main', 'GLI', 'Ours', 5, 'DSC'): (94.8, 3.9),
 ('main', 'GLI', 'Ours', 6, 'DSC'): (98.4, 2.9),
 ('main', 'GLI', 'Ours', 7, 'DSC'): (97.2, 3.4),
 ('main', 'GLI', 'Ours', 1, 'HD95'): (1.07, 0.29),
 ('main', 'GLI', 'Ours', 2, 'HD95'): (1.22, 0.66),
 ('main', 'GLI', 'Ours', 3, 'HD95'): (1.1, 0.36),
 ('main', 'GLI', 'Ours', 4, 'HD95'): (9.46, 12.46),
 ('main', 'GLI', 'Ours', 5, 'HD95'): (1.26, 0.84),
 ('main', 'GLI', 'Ours', 6, 'HD95'): (1.21, 1.89),
 ('main', 'GLI', 'Ours', 7, 'HD95'): (1.22, 1.7),
 ('main', 'GLI', 'Joint Upper Bound', 4, 'DSC'): (92.6, 6.7),
 ('main', 'GLI', 'Joint Upper Bound', 4, 'HD95'): (3.83, 5.3),
 ('main', 'WMH', 'Lesion-Only', 4, 'DSC'): (3.3, 11.5),
 ('main', 'WMH', 'Lesion-Only', 4, 'HD95'): (299.9, 130.9),
 ('main', 'WMH', 'BrainSeg', 4, 'DSC'): (19.4, 19.0),
 ('main', 'WMH', 'BrainSeg', 4, 'HD95'): (36.51, 14.38),
 ('main', 'WMH', 'Joint-Baseline', 1, 'DSC'): (79.5, 2.1),
 ('main', 'WMH', 'Joint-Baseline', 2, 'DSC'): (88.4, 2.0),
 ('main', 'WMH', 'Joint-Baseline', 3, 'DSC'): (89.5, 3.4),
 ('main', 'WMH', 'Joint-Baseline', 4, 'DSC'): (3.6, 11.9),
 ('main', 'WMH', 'Joint-Baseline', 5, 'DSC'): (89.6, 3.4),
 ('main', 'WMH', 'Joint-Baseline', 6, 'DSC'): (91.4, 1.2),
 ('main', 'WMH', 'Joint-Baseline', 7, 'DSC'): (87.2, 2.2),
 ('main', 'WMH', 'Joint-Baseline', 1, 'HD95'): (6.35, 1.26),
 ('main', 'WMH', 'Joint-Baseline', 2, 'HD95'): (1.68, 0.34),
 ('main', 'WMH', 'Joint-Baseline', 3, 'HD95'): (1.9, 0.66),
 ('main', 'WMH', 'Joint-Baseline', 4, 'HD95'): (303.43, 127.92),
 ('main', 'WMH', 'Joint-Baseline', 5, 'HD95'): (1.76, 0.34),
 ('main', 'WMH', 'Joint-Baseline', 6, 'HD95'): (4.85, 0.92),
 ('main', 'WMH', 'Joint-Baseline', 7, 'HD95'): (2.31, 0.5),
 ('main', 'WMH', 'TumorSynth', 1, 'DSC'): (78.4, 2.3),
 ('main', 'WMH', 'TumorSynth', 2, 'DSC'): (87.6, 1.9),
 ('main', 'WMH', 'TumorSynth', 3, 'DSC'): (88.2, 3.9),
 ('main', 'WMH', 'TumorSynth', 4, 'DSC'): (16.7, 19.8),
 ('main', 'WMH', 'TumorSynth', 5, 'DSC'): (87.4, 4.9),
 ('main', 'WMH', 'TumorSynth', 6, 'DSC'): (91.6, 1.4),
 ('main', 'WMH', 'TumorSynth', 7, 'DSC'): (85.2, 3.1),
 ('main', 'WMH', 'TumorSynth', 1, 'HD95'): (5.77, 1.09),
 ('main', 'WMH', 'TumorSynth', 2, 'HD95'): (1.79, 0.32),
 ('main', 'WMH', 'TumorSynth', 3, 'HD95'): (1.99, 0.54),
 ('main', 'WMH', 'TumorSynth', 4, 'HD95'): (162.59, 152.5),
 ('main', 'WMH', 'TumorSynth', 5, 'HD95'): (1.98, 0.36),
 ('main', 'WMH', 'TumorSynth', 6, 'HD95'): (4.65, 0.87),
 ('main', 'WMH', 'TumorSynth', 7, 'HD95'): (2.64, 0.69),
 ('main', 'WMH', 'UNA', 1, 'DSC'): (79.3, 2.2),
 ('main', 'WMH', 'UNA', 2, 'DSC'): (88.4, 2.0),
 ('main', 'WMH', 'UNA', 3, 'DSC'): (90.1, 2.7),
 ('main', 'WMH', 'UNA', 4, 'DSC'): (31.1, 28.9),
 ('main', 'WMH', 'UNA', 5, 'DSC'): (89.1, 3.4),
 ('main', 'WMH', 'UNA', 6, 'DSC'): (91.6, 1.1),
 ('main', 'WMH', 'UNA', 7, 'DSC'): (87.4, 2.0),
 ('main', 'WMH', 'UNA', 1, 'HD95'): (6.39, 1.26),
 ('main', 'WMH', 'UNA', 2, 'HD95'): (1.68, 0.33),
 ('main', 'WMH', 'UNA', 3, 'HD95'): (1.59, 0.33),
 ('main', 'WMH', 'UNA', 4, 'HD95'): (106.06, 128.86),
 ('main', 'WMH', 'UNA', 5, 'HD95'): (1.84, 0.36),
 ('main', 'WMH', 'UNA', 6, 'HD95'): (4.85, 0.89),
 ('main', 'WMH', 'UNA', 7, 'HD95'): (2.28, 0.47),
 ('main', 'WMH', 'MedLesSynth-LD', 1, 'DSC'): (78.3, 2.1),
 ('main', 'WMH', 'MedLesSynth-LD', 2, 'DSC'): (88.9, 1.8),
 ('main', 'WMH', 'MedLesSynth-LD', 3, 'DSC'): (90.2, 2.7),
 ('main', 'WMH', 'MedLesSynth-LD', 4, 'DSC'): (33.8, 28.3),
 ('main', 'WMH', 'MedLesSynth-LD', 5, 'DSC'): (89.8, 3.3),
 ('main', 'WMH', 'MedLesSynth-LD', 6, 'DSC'): (92.0, 1.2),
 ('main', 'WMH', 'MedLesSynth-LD', 7, 'DSC'): (87.7, 2.1),
 ('main', 'WMH', 'MedLesSynth-LD', 1, 'HD95'): (6.4, 1.29),
 ('main', 'WMH', 'MedLesSynth-LD', 2, 'HD95'): (1.64, 0.31),
 ('main', 'WMH', 'MedLesSynth-LD', 3, 'HD95'): (1.56, 0.3),
 ('main', 'WMH', 'MedLesSynth-LD', 4, 'HD95'): (73.36, 98.41),
 ('main', 'WMH', 'MedLesSynth-LD', 5, 'HD95'): (1.74, 0.34),
 ('main', 'WMH', 'MedLesSynth-LD', 6, 'HD95'): (4.7, 0.93),
 ('main', 'WMH', 'MedLesSynth-LD', 7, 'HD95'): (2.23, 0.48),
 ('main', 'WMH', 'Ours', 1, 'DSC'): (79.4, 2.1),
 ('main', 'WMH', 'Ours', 2, 'DSC'): (88.4, 1.9),
 ('main', 'WMH', 'Ours', 3, 'DSC'): (90.1, 2.7),
 ('main', 'WMH', 'Ours', 4, 'DSC'): (40.1, 25.3),
 ('main', 'WMH', 'Ours', 5, 'DSC'): (89.0, 3.4),
 ('main', 'WMH', 'Ours', 6, 'DSC'): (91.4, 1.2),
 ('main', 'WMH', 'Ours', 7, 'DSC'): (87.5, 2.0),
 ('main', 'WMH', 'Ours', 1, 'HD95'): (6.38, 1.26),
 ('main', 'WMH', 'Ours', 2, 'HD95'): (1.68, 0.36),
 ('main', 'WMH', 'Ours', 3, 'HD95'): (1.5, 0.27),
 ('main', 'WMH', 'Ours', 4, 'HD95'): (22.69, 12.18),
 ('main', 'WMH', 'Ours', 5, 'HD95'): (1.84, 0.35),
 ('main', 'WMH', 'Ours', 6, 'HD95'): (4.85, 0.85),
 ('main', 'WMH', 'Ours', 7, 'HD95'): (2.26, 0.43),
 ('main', 'WMH', 'Joint Upper Bound', 4, 'DSC'): (66.9, 13.8),
 ('main', 'WMH', 'Joint Upper Bound', 4, 'HD95'): (8.47, 8.57),
 ('main', 'MEN', 'Lesion-Only', 4, 'DSC'): (55.5, 39.8),
 ('main', 'MEN', 'Lesion-Only', 4, 'HD95'): (111.53, 146.88),
 ('main', 'MEN', 'BrainSeg', 4, 'DSC'): (35.1, 31.3),
 ('main', 'MEN', 'BrainSeg', 4, 'HD95'): (42.02, 22.12),
 ('main', 'MEN', 'Joint-Baseline', 1, 'DSC'): (95.9, 1.5),
 ('main', 'MEN', 'Joint-Baseline', 2, 'DSC'): (96.4, 2.9),
 ('main', 'MEN', 'Joint-Baseline', 3, 'DSC'): (95.9, 2.0),
 ('main', 'MEN', 'Joint-Baseline', 4, 'DSC'): (56.3, 39.4),
 ('main', 'MEN', 'Joint-Baseline', 5, 'DSC'): (96.1, 2.2),
 ('main', 'MEN', 'Joint-Baseline', 6, 'DSC'): (98.3, 1.6),
 ('main', 'MEN', 'Joint-Baseline', 7, 'DSC'): (97.6, 1.5),
 ('main', 'MEN', 'Joint-Baseline', 1, 'HD95'): (1.03, 0.13),
 ('main', 'MEN', 'Joint-Baseline', 2, 'HD95'): (1.06, 0.29),
 ('main', 'MEN', 'Joint-Baseline', 3, 'HD95'): (1.09, 0.3),
 ('main', 'MEN', 'Joint-Baseline', 4, 'HD95'): (114.3, 150.67),
 ('main', 'MEN', 'Joint-Baseline', 5, 'HD95'): (1.04, 0.18),
 ('main', 'MEN', 'Joint-Baseline', 6, 'HD95'): (1.22, 0.85),
 ('main', 'MEN', 'Joint-Baseline', 7, 'HD95'): (1.06, 0.25),
 ('main', 'MEN', 'TumorSynth', 1, 'DSC'): (92.5, 1.7),
 ('main', 'MEN', 'TumorSynth', 2, 'DSC'): (94.6, 3.2),
 ('main', 'MEN', 'TumorSynth', 3, 'DSC'): (93.1, 2.0),
 ('main', 'MEN', 'TumorSynth', 4, 'DSC'): (49.8, 39.6),
 ('main', 'MEN', 'TumorSynth', 5, 'DSC'): (93.7, 3.1),
 ('main', 'MEN', 'TumorSynth', 6, 'DSC'): (97.6, 2.5),
 ('main', 'MEN', 'TumorSynth', 7, 'DSC'): (96.5, 1.8),
 ('main', 'MEN', 'TumorSynth', 1, 'HD95'): (1.25, 0.29),
 ('main', 'MEN', 'TumorSynth', 2, 'HD95'): (1.1, 0.34),
 ('main', 'MEN', 'TumorSynth', 3, 'HD95'): (1.46, 0.47),
 ('main', 'MEN', 'TumorSynth', 4, 'HD95'): (117.97, 149.03),
 ('main', 'MEN', 'TumorSynth', 5, 'HD95'): (1.09, 0.34),
 ('main', 'MEN', 'TumorSynth', 6, 'HD95'): (1.39, 1.39),
 ('main', 'MEN', 'TumorSynth', 7, 'HD95'): (1.13, 0.56),
 ('main', 'MEN', 'jSTABL', 1, 'DSC'): (80.7, 4.8),
 ('main', 'MEN', 'jSTABL', 2, 'DSC'): (66.9, 6.0),
 ('main', 'MEN', 'jSTABL', 3, 'DSC'): (84.6, 5.0),
 ('main', 'MEN', 'jSTABL', 4, 'DSC'): (58.2, 38.9),
 ('main', 'MEN', 'jSTABL', 5, 'DSC'): (81.7, 8.6),
 ('main', 'MEN', 'jSTABL', 6, 'DSC'): (92.9, 2.5),
 ('main', 'MEN', 'jSTABL', 7, 'DSC'): (71.2, 2.4),
 ('main', 'MEN', 'jSTABL', 1, 'HD95'): (2.41, 0.53),
 ('main', 'MEN', 'jSTABL', 2, 'HD95'): (11.88, 2.37),
 ('main', 'MEN', 'jSTABL', 3, 'HD95'): (2.95, 1.39),
 ('main', 'MEN', 'jSTABL', 4, 'HD95'): (70.68, 111.27),
 ('main', 'MEN', 'jSTABL', 5, 'HD95'): (5.51, 2.88),
 ('main', 'MEN', 'jSTABL', 6, 'HD95'): (3.17, 1.11),
 ('main', 'MEN', 'jSTABL', 7, 'HD95'): (13.45, 1.09),
 ('main', 'MEN', 'UNA', 1, 'DSC'): (95.8, 1.6),
 ('main', 'MEN', 'UNA', 2, 'DSC'): (96.0, 3.0),
 ('main', 'MEN', 'UNA', 3, 'DSC'): (95.8, 2.0),
 ('main', 'MEN', 'UNA', 4, 'DSC'): (57.0, 37.7),
 ('main', 'MEN', 'UNA', 5, 'DSC'): (95.7, 2.4),
 ('main', 'MEN', 'UNA', 6, 'DSC'): (98.1, 1.9),
 ('main', 'MEN', 'UNA', 7, 'DSC'): (97.2, 2.4),
 ('main', 'MEN', 'UNA', 1, 'HD95'): (1.04, 0.15),
 ('main', 'MEN', 'UNA', 2, 'HD95'): (1.07, 0.29),
 ('main', 'MEN', 'UNA', 3, 'HD95'): (1.09, 0.28),
 ('main', 'MEN', 'UNA', 4, 'HD95'): (86.16, 130.93),
 ('main', 'MEN', 'UNA', 5, 'HD95'): (1.06, 0.24),
 ('main', 'MEN', 'UNA', 6, 'HD95'): (1.27, 0.9),
 ('main', 'MEN', 'UNA', 7, 'HD95'): (1.13, 0.62),
 ('main', 'MEN', 'MedLesSynth-LD', 1, 'DSC'): (95.8, 1.6),
 ('main', 'MEN', 'MedLesSynth-LD', 2, 'DSC'): (96.1, 3.0),
 ('main', 'MEN', 'MedLesSynth-LD', 3, 'DSC'): (95.8, 2.0),
 ('main', 'MEN', 'MedLesSynth-LD', 4, 'DSC'): (57.0, 38.2),
 ('main', 'MEN', 'MedLesSynth-LD', 5, 'DSC'): (95.8, 2.4),
 ('main', 'MEN', 'MedLesSynth-LD', 6, 'DSC'): (98.2, 1.7),
 ('main', 'MEN', 'MedLesSynth-LD', 7, 'DSC'): (97.3, 1.7),
 ('main', 'MEN', 'MedLesSynth-LD', 1, 'HD95'): (1.04, 0.16),
 ('main', 'MEN', 'MedLesSynth-LD', 2, 'HD95'): (1.07, 0.29),
 ('main', 'MEN', 'MedLesSynth-LD', 3, 'HD95'): (1.09, 0.28),
 ('main', 'MEN', 'MedLesSynth-LD', 4, 'HD95'): (89.19, 132.39),
 ('main', 'MEN', 'MedLesSynth-LD', 5, 'HD95'): (1.06, 0.24),
 ('main', 'MEN', 'MedLesSynth-LD', 6, 'HD95'): (1.2, 0.72),
 ('main', 'MEN', 'MedLesSynth-LD', 7, 'HD95'): (1.09, 0.32),
 ('main', 'MEN', 'Ours', 1, 'DSC'): (95.8, 1.6),
 ('main', 'MEN', 'Ours', 2, 'DSC'): (96.0, 3.0),
 ('main', 'MEN', 'Ours', 3, 'DSC'): (95.8, 2.0),
 ('main', 'MEN', 'Ours', 4, 'DSC'): (58.9, 36.1),
 ('main', 'MEN', 'Ours', 5, 'DSC'): (95.7, 2.4),
 ('main', 'MEN', 'Ours', 6, 'DSC'): (98.1, 1.8),
 ('main', 'MEN', 'Ours', 7, 'DSC'): (97.3, 1.8),
 ('main', 'MEN', 'Ours', 1, 'HD95'): (1.04, 0.15),
 ('main', 'MEN', 'Ours', 2, 'HD95'): (1.08, 0.31),
 ('main', 'MEN', 'Ours', 3, 'HD95'): (1.09, 0.27),
 ('main', 'MEN', 'Ours', 4, 'HD95'): (42.22, 73.6),
 ('main', 'MEN', 'Ours', 5, 'HD95'): (1.07, 0.25),
 ('main', 'MEN', 'Ours', 6, 'HD95'): (1.24, 0.81),
 ('main', 'MEN', 'Ours', 7, 'HD95'): (1.1, 0.37),
 ('main', 'MEN', 'Joint Upper Bound', 4, 'DSC'): (84.9, 18.9),
 ('main', 'MEN', 'Joint Upper Bound', 4, 'HD95'): (10.19, 31.67),
 ('main', 'MRBrainS18', 'Lesion-Only', 4, 'DSC'): (4.9, 19.9),
 ('main', 'MRBrainS18', 'Lesion-Only', 4, 'HD95'): (332.15, 109.39),
 ('main', 'MRBrainS18', 'BrainSeg', 4, 'DSC'): (14.3, 15.2),
 ('main', 'MRBrainS18', 'BrainSeg', 4, 'HD95'): (49.26, 62.99),
 ('main', 'MRBrainS18', 'Joint-Baseline', 1, 'DSC'): (72.1, 2.3),
 ('main', 'MRBrainS18', 'Joint-Baseline', 2, 'DSC'): (65.0, 2.8),
 ('main', 'MRBrainS18', 'Joint-Baseline', 3, 'DSC'): (79.7, 3.6),
 ('main', 'MRBrainS18', 'Joint-Baseline', 4, 'DSC'): (5.2, 20.2),
 ('main', 'MRBrainS18', 'Joint-Baseline', 5, 'DSC'): (86.1, 3.6),
 ('main', 'MRBrainS18', 'Joint-Baseline', 6, 'DSC'): (88.6, 2.5),
 ('main', 'MRBrainS18', 'Joint-Baseline', 7, 'DSC'): (75.3, 3.3),
 ('main', 'MRBrainS18', 'Joint-Baseline', 1, 'HD95'): (9.32, 1.6),
 ('main', 'MRBrainS18', 'Joint-Baseline', 2, 'HD95'): (12.27, 1.41),
 ('main', 'MRBrainS18', 'Joint-Baseline', 3, 'HD95'): (2.63, 0.46),
 ('main', 'MRBrainS18', 'Joint-Baseline', 4, 'HD95'): (331.8, 110.17),
 ('main', 'MRBrainS18', 'Joint-Baseline', 5, 'HD95'): (3.36, 1.11),
 ('main', 'MRBrainS18', 'Joint-Baseline', 6, 'HD95'): (6.01, 1.15),
 ('main', 'MRBrainS18', 'Joint-Baseline', 7, 'HD95'): (7.33, 1.33),
 ('main', 'MRBrainS18', 'TumorSynth', 1, 'DSC'): (70.5, 2.5),
 ('main', 'MRBrainS18', 'TumorSynth', 2, 'DSC'): (65.1, 2.8),
 ('main', 'MRBrainS18', 'TumorSynth', 3, 'DSC'): (78.6, 3.7),
 ('main', 'MRBrainS18', 'TumorSynth', 4, 'DSC'): (13.7, 23.3),
 ('main', 'MRBrainS18', 'TumorSynth', 5, 'DSC'): (84.0, 4.8),
 ('main', 'MRBrainS18', 'TumorSynth', 6, 'DSC'): (88.6, 2.5),
 ('main', 'MRBrainS18', 'TumorSynth', 7, 'DSC'): (74.9, 3.0),
 ('main', 'MRBrainS18', 'TumorSynth', 1, 'HD95'): (8.85, 1.45),
 ('main', 'MRBrainS18', 'TumorSynth', 2, 'HD95'): (12.08, 1.45),
 ('main', 'MRBrainS18', 'TumorSynth', 3, 'HD95'): (2.76, 0.36),
 ('main', 'MRBrainS18', 'TumorSynth', 4, 'HD95'): (206.62, 160.52),
 ('main', 'MRBrainS18', 'TumorSynth', 5, 'HD95'): (3.67, 1.35),
 ('main', 'MRBrainS18', 'TumorSynth', 6, 'HD95'): (5.79, 1.06),
 ('main', 'MRBrainS18', 'TumorSynth', 7, 'HD95'): (7.25, 1.28),
 ('main', 'MRBrainS18', 'UNA', 1, 'DSC'): (72.1, 2.4),
 ('main', 'MRBrainS18', 'UNA', 2, 'DSC'): (65.1, 2.7),
 ('main', 'MRBrainS18', 'UNA', 3, 'DSC'): (80.2, 3.2),
 ('main', 'MRBrainS18', 'UNA', 4, 'DSC'): (22.1, 29.0),
 ('main', 'MRBrainS18', 'UNA', 5, 'DSC'): (86.2, 3.6),
 ('main', 'MRBrainS18', 'UNA', 6, 'DSC'): (88.8, 2.5),
 ('main', 'MRBrainS18', 'UNA', 7, 'DSC'): (75.2, 3.3),
 ('main', 'MRBrainS18', 'UNA', 1, 'HD95'): (9.35, 1.59),
 ('main', 'MRBrainS18', 'UNA', 2, 'HD95'): (12.26, 1.41),
 ('main', 'MRBrainS18', 'UNA', 3, 'HD95'): (2.48, 0.33),
 ('main', 'MRBrainS18', 'UNA', 4, 'HD95'): (166.12, 162.05),
 ('main', 'MRBrainS18', 'UNA', 5, 'HD95'): (3.35, 1.11),
 ('main', 'MRBrainS18', 'UNA', 6, 'HD95'): (6.01, 1.19),
 ('main', 'MRBrainS18', 'UNA', 7, 'HD95'): (7.38, 1.35),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 1, 'DSC'): (71.3, 2.4),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 2, 'DSC'): (65.3, 2.7),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 3, 'DSC'): (80.4, 3.2),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 4, 'DSC'): (20.8, 24.8),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 5, 'DSC'): (86.3, 3.4),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 6, 'DSC'): (89.2, 2.5),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 7, 'DSC'): (75.5, 3.3),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 1, 'HD95'): (9.36, 1.59),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 2, 'HD95'): (12.2, 1.4),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 3, 'HD95'): (2.43, 0.3),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 4, 'HD95'): (61.74, 86.52),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 5, 'HD95'): (3.27, 0.98),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 6, 'HD95'): (5.94, 1.18),
 ('main', 'MRBrainS18', 'MedLesSynth-LD', 7, 'HD95'): (7.32, 1.36),
 ('main', 'MRBrainS18', 'Ours', 1, 'DSC'): (72.1, 2.3),
 ('main', 'MRBrainS18', 'Ours', 2, 'DSC'): (65.0, 2.7),
 ('main', 'MRBrainS18', 'Ours', 3, 'DSC'): (80.2, 3.2),
 ('main', 'MRBrainS18', 'Ours', 4, 'DSC'): (25.4, 23.3),
 ('main', 'MRBrainS18', 'Ours', 5, 'DSC'): (86.1, 3.6),
 ('main', 'MRBrainS18', 'Ours', 6, 'DSC'): (88.7, 2.5),
 ('main', 'MRBrainS18', 'Ours', 7, 'DSC'): (75.2, 3.3),
 ('main', 'MRBrainS18', 'Ours', 1, 'HD95'): (9.34, 1.58),
 ('main', 'MRBrainS18', 'Ours', 2, 'HD95'): (12.26, 1.4),
 ('main', 'MRBrainS18', 'Ours', 3, 'HD95'): (2.37, 0.24),
 ('main', 'MRBrainS18', 'Ours', 4, 'HD95'): (40.49, 64.03),
 ('main', 'MRBrainS18', 'Ours', 5, 'HD95'): (3.41, 1.1),
 ('main', 'MRBrainS18', 'Ours', 6, 'HD95'): (6.01, 1.19),
 ('main', 'MRBrainS18', 'Ours', 7, 'HD95'): (7.36, 1.35),
 ('main', 'MRBrainS18', 'Joint Upper Bound', 4, 'DSC'): (53.8, 25.9),
 ('main', 'MRBrainS18', 'Joint Upper Bound', 4, 'HD95'): (23.58, 66.92),
 ('main', 'Stroke', 'Lesion-Only', 4, 'DSC'): (29.2, 27.9),
 ('main', 'Stroke', 'BrainSeg', 4, 'DSC'): (25.8, 20.1),
 ('main', 'Stroke', 'Joint-Baseline', 4, 'DSC'): (31.9, 27.9),
 ('main', 'Stroke', 'TumorSynth', 4, 'DSC'): (24.0, 24.3),
 ('main', 'Stroke', 'UNA', 4, 'DSC'): (45.0, 26.9),
 ('main', 'Stroke', 'MedLesSynth-LD', 4, 'DSC'): (49.4, 25.7),
 ('main', 'Stroke', 'Ours', 4, 'DSC'): (50.6, 24.5),
 ('main', 'Stroke', 'Joint Upper Bound', 4, 'DSC'): (77.5, 16.5),
 ('main', 'Stroke', 'Lesion-Only', 4, 'HD95'): (125.72, 136.9),
 ('main', 'Stroke', 'BrainSeg', 4, 'HD95'): (28.40, 14.23),
 ('main', 'Stroke', 'Joint-Baseline', 4, 'HD95'): (108.88, 129.85),
 ('main', 'Stroke', 'TumorSynth', 4, 'HD95'): (124.06, 139.42),
 ('main', 'Stroke', 'UNA', 4, 'HD95'): (44.83, 56.82),
 ('main', 'Stroke', 'MedLesSynth-LD', 4, 'HD95'): (34.73, 41.05),
 ('main', 'Stroke', 'Ours', 4, 'HD95'): (18.9, 10.81),
 ('main', 'Stroke', 'Joint Upper Bound', 4, 'HD95'): (7.26, 8.38),
 ('main', 'MET', 'Lesion-Only', 4, 'DSC'): (59.7, 36.8),
 ('main', 'MET', 'BrainSeg', 4, 'DSC'): (50.0, 32.4),
 ('main', 'MET', 'Joint-Baseline', 4, 'DSC'): (60.7, 36.0),
 ('main', 'MET', 'TumorSynth', 4, 'DSC'): (43.1, 39.2),
 ('main', 'MET', 'jSTABL', 4, 'DSC'): (60.7, 33.6),
 ('main', 'MET', 'UNA', 4, 'DSC'): (64.2, 33.4),
 ('main', 'MET', 'MedLesSynth-LD', 4, 'DSC'): (63.9, 33.1),
 ('main', 'MET', 'Ours', 4, 'DSC'): (65.9, 31.7),
 ('main', 'MET', 'Joint Upper Bound', 4, 'DSC'): (76.7, 22.3),
 ('main', 'MET', 'Lesion-Only', 4, 'HD95'): (99.36, 136.51),
 ('main', 'MET', 'BrainSeg', 4, 'HD95'): (32.14, 21.52),
 ('main', 'MET', 'Joint-Baseline', 4, 'HD95'): (85.01, 124.33),
 ('main', 'MET', 'TumorSynth', 4, 'HD95'): (131.17, 152.25),
 ('main', 'MET', 'jSTABL', 4, 'HD95'): (57.09, 88.31),
 ('main', 'MET', 'UNA', 4, 'HD95'): (63.96, 100.11),
 ('main', 'MET', 'MedLesSynth-LD', 4, 'HD95'): (55.11, 89.37),
 ('main', 'MET', 'Ours', 4, 'HD95'): (35.4, 56.93),
 ('main', 'MET', 'Joint Upper Bound', 4, 'HD95'): (13.98, 23.43),
 ('main', 'MS', 'Lesion-Only', 4, 'DSC'): (5.2, 9.1),
 ('main', 'MS', 'BrainSeg', 4, 'DSC'): (12.0, 10.6),
 ('main', 'MS', 'Joint-Baseline', 4, 'DSC'): (15.0, 16.5),
 ('main', 'MS', 'TumorSynth', 4, 'DSC'): (5.0, 7.3),
 ('main', 'MS', 'UNA', 4, 'DSC'): (22.6, 22.0),
 ('main', 'MS', 'MedLesSynth-LD', 4, 'DSC'): (32.9, 19.0),
 ('main', 'MS', 'Ours', 4, 'DSC'): (40.4, 16.8),
 ('main', 'MS', 'Joint Upper Bound', 4, 'DSC'): (67.1, 8.2),
 ('main', 'MS', 'Lesion-Only', 4, 'HD95'): (254.05, 148.11),
 ('main', 'MS', 'BrainSeg', 4, 'HD95'): (43.52, 17.07),
 ('main', 'MS', 'Joint-Baseline', 4, 'HD95'): (130.36, 118.71),
 ('main', 'MS', 'TumorSynth', 4, 'HD95'): (200.39, 148.99),
 ('main', 'MS', 'UNA', 4, 'HD95'): (78.91, 97.84),
 ('main', 'MS', 'MedLesSynth-LD', 4, 'HD95'): (49.03, 18.08),
 ('main', 'MS', 'Ours', 4, 'HD95'): (31.24, 14.71),
 ('main', 'MS', 'Joint Upper Bound', 4, 'HD95'): (16.3, 12.4),
 ('main', 'PED', 'Lesion-Only', 4, 'DSC'): (89.5, 6.1),
 ('main', 'PED', 'BrainSeg', 4, 'DSC'): (63.2, 17.0),
 ('main', 'PED', 'Joint-Baseline', 4, 'DSC'): (90.7, 4.0),
 ('main', 'PED', 'TumorSynth', 4, 'DSC'): (54.5, 35.7),
 ('main', 'PED', 'jSTABL', 4, 'DSC'): (55.0, 34.1),
 ('main', 'PED', 'UNA', 4, 'DSC'): (90.1, 3.7),
 ('main', 'PED', 'MedLesSynth-LD', 4, 'DSC'): (90.3, 4.2),
 ('main', 'PED', 'Ours', 4, 'DSC'): (90.2, 3.8),
 ('main', 'PED', 'Joint Upper Bound', 4, 'DSC'): (91.2, 3.9),
 ('main', 'PED', 'Lesion-Only', 4, 'HD95'): (6.85, 10.81),
 ('main', 'PED', 'BrainSeg', 4, 'HD95'): (29.93, 17.91),
 ('main', 'PED', 'Joint-Baseline', 4, 'HD95'): (5.98, 10.66),
 ('main', 'PED', 'TumorSynth', 4, 'HD95'): (36.22, 81.58),
 ('main', 'PED', 'jSTABL', 4, 'HD95'): (28.2, 30.41),
 ('main', 'PED', 'UNA', 4, 'HD95'): (6.29, 10.62),
 ('main', 'PED', 'MedLesSynth-LD', 4, 'HD95'): (6.71, 10.88),
 ('main', 'PED', 'Ours', 4, 'HD95'): (6.09, 10.61),
 ('main', 'PED', 'Joint Upper Bound', 4, 'HD95'): (6.98, 11.09),
 ('main', 'SSA', 'Lesion-Only', 4, 'DSC'): (93.8, 2.1),
 ('main', 'SSA', 'BrainSeg', 4, 'DSC'): (71.1, 13.0),
 ('main', 'SSA', 'Joint-Baseline', 4, 'DSC'): (94.1, 2.1),
 ('main', 'SSA', 'TumorSynth', 4, 'DSC'): (87.4, 4.6),
 ('main', 'SSA', 'jSTABL', 4, 'DSC'): (92.0, 3.7),
 ('main', 'SSA', 'UNA', 4, 'DSC'): (93.0, 3.2),
 ('main', 'SSA', 'MedLesSynth-LD', 4, 'DSC'): (93.0, 2.8),
 ('main', 'SSA', 'Ours', 4, 'DSC'): (93.3, 2.1),
 ('main', 'SSA', 'Joint Upper Bound', 4, 'DSC'): (94.1, 3.0),
 ('main', 'SSA', 'Lesion-Only', 4, 'HD95'): (9.22, 14.58),
 ('main', 'SSA', 'BrainSeg', 4, 'HD95'): (17.61, 14.51),
 ('main', 'SSA', 'Joint-Baseline', 4, 'HD95'): (11.71, 17.1),
 ('main', 'SSA', 'TumorSynth', 4, 'HD95'): (14.66, 16.38),
 ('main', 'SSA', 'jSTABL', 4, 'HD95'): (11.89, 14.97),
 ('main', 'SSA', 'UNA', 4, 'HD95'): (6.6, 6.15),
 ('main', 'SSA', 'MedLesSynth-LD', 4, 'HD95'): (7.07, 6.37),
 ('main', 'SSA', 'Ours', 4, 'HD95'): (8.93, 11.88),
 ('main', 'SSA', 'Joint Upper Bound', 4, 'HD95'): (4.09, 2.4),
 ('ablation', 'GLI', 'Baseline', 1, 'DSC'): (96.0, 1.9),
 ('ablation', 'GLI', 'Baseline', 2, 'DSC'): (95.2, 3.9),
 ('ablation', 'GLI', 'Baseline', 3, 'DSC'): (95.9, 2.3),
 ('ablation', 'GLI', 'Baseline', 4, 'DSC'): (91.0, 9.4),
 ('ablation', 'GLI', 'Baseline', 5, 'DSC'): (94.9, 3.8),
 ('ablation', 'GLI', 'Baseline', 6, 'DSC'): (98.4, 2.9),
 ('ablation', 'GLI', 'Baseline', 7, 'DSC'): (97.2, 3.4),
 ('ablation', 'GLI', 'Baseline', 1, 'HD95'): (1.07, 0.29),
 ('ablation', 'GLI', 'Baseline', 2, 'HD95'): (1.21, 0.64),
 ('ablation', 'GLI', 'Baseline', 3, 'HD95'): (1.1, 0.36),
 ('ablation', 'GLI', 'Baseline', 4, 'HD95'): (13.04, 17.22),
 ('ablation', 'GLI', 'Baseline', 5, 'HD95'): (1.25, 0.84),
 ('ablation', 'GLI', 'Baseline', 6, 'HD95'): (1.21, 1.89),
 ('ablation', 'GLI', 'Baseline', 7, 'HD95'): (1.22, 1.7),
 ('ablation', 'GLI', 'w/o Simplex', 1, 'DSC'): (96.0, 1.9),
 ('ablation', 'GLI', 'w/o Simplex', 2, 'DSC'): (95.1, 4.0),
 ('ablation', 'GLI', 'w/o Simplex', 3, 'DSC'): (95.9, 2.3),
 ('ablation', 'GLI', 'w/o Simplex', 4, 'DSC'): (91.2, 8.8),
 ('ablation', 'GLI', 'w/o Simplex', 5, 'DSC'): (94.8, 3.8),
 ('ablation', 'GLI', 'w/o Simplex', 6, 'DSC'): (98.4, 2.9),
 ('ablation', 'GLI', 'w/o Simplex', 7, 'DSC'): (97.2, 3.4),
 ('ablation', 'GLI', 'w/o Simplex', 1, 'HD95'): (1.06, 0.28),
 ('ablation', 'GLI', 'w/o Simplex', 2, 'HD95'): (1.22, 0.66),
 ('ablation', 'GLI', 'w/o Simplex', 3, 'HD95'): (1.11, 0.36),
 ('ablation', 'GLI', 'w/o Simplex', 4, 'HD95'): (10.2, 13.23),
 ('ablation', 'GLI', 'w/o Simplex', 5, 'HD95'): (1.26, 0.81),
 ('ablation', 'GLI', 'w/o Simplex', 6, 'HD95'): (1.22, 1.89),
 ('ablation', 'GLI', 'w/o Simplex', 7, 'HD95'): (1.22, 1.7),
 ('ablation', 'GLI', 'w/o FDAR', 1, 'DSC'): (96.0, 1.9),
 ('ablation', 'GLI', 'w/o FDAR', 2, 'DSC'): (95.1, 4.0),
 ('ablation', 'GLI', 'w/o FDAR', 3, 'DSC'): (95.9, 2.3),
 ('ablation', 'GLI', 'w/o FDAR', 4, 'DSC'): (91.2, 8.8),
 ('ablation', 'GLI', 'w/o FDAR', 5, 'DSC'): (94.9, 3.8),
 ('ablation', 'GLI', 'w/o FDAR', 6, 'DSC'): (98.4, 2.9),
 ('ablation', 'GLI', 'w/o FDAR', 7, 'DSC'): (97.2, 3.4),
 ('ablation', 'GLI', 'w/o FDAR', 1, 'HD95'): (1.07, 0.29),
 ('ablation', 'GLI', 'w/o FDAR', 2, 'HD95'): (1.21, 0.65),
 ('ablation', 'GLI', 'w/o FDAR', 3, 'HD95'): (1.1, 0.36),
 ('ablation', 'GLI', 'w/o FDAR', 4, 'HD95'): (11.87, 16.03),
 ('ablation', 'GLI', 'w/o FDAR', 5, 'HD95'): (1.25, 0.8),
 ('ablation', 'GLI', 'w/o FDAR', 6, 'HD95'): (1.22, 1.89),
 ('ablation', 'GLI', 'w/o FDAR', 7, 'HD95'): (1.22, 1.7),
 ('ablation', 'GLI', 'w/o MIPB', 1, 'DSC'): (96.0, 1.9),
 ('ablation', 'GLI', 'w/o MIPB', 2, 'DSC'): (95.1, 4.0),
 ('ablation', 'GLI', 'w/o MIPB', 3, 'DSC'): (95.9, 2.3),
 ('ablation', 'GLI', 'w/o MIPB', 4, 'DSC'): (91.2, 8.8),
 ('ablation', 'GLI', 'w/o MIPB', 5, 'DSC'): (94.9, 3.6),
 ('ablation', 'GLI', 'w/o MIPB', 6, 'DSC'): (98.4, 2.9),
 ('ablation', 'GLI', 'w/o MIPB', 7, 'DSC'): (97.2, 3.4),
 ('ablation', 'GLI', 'w/o MIPB', 1, 'HD95'): (1.06, 0.28),
 ('ablation', 'GLI', 'w/o MIPB', 2, 'HD95'): (1.22, 0.67),
 ('ablation', 'GLI', 'w/o MIPB', 3, 'HD95'): (1.1, 0.36),
 ('ablation', 'GLI', 'w/o MIPB', 4, 'HD95'): (12.91, 17.0),
 ('ablation', 'GLI', 'w/o MIPB', 5, 'HD95'): (1.25, 0.75),
 ('ablation', 'GLI', 'w/o MIPB', 6, 'HD95'): (1.22, 1.89),
 ('ablation', 'GLI', 'w/o MIPB', 7, 'HD95'): (1.22, 1.7),
 ('ablation', 'GLI', 'Ours', 1, 'DSC'): (96.0, 1.9),
 ('ablation', 'GLI', 'Ours', 2, 'DSC'): (95.1, 4.0),
 ('ablation', 'GLI', 'Ours', 3, 'DSC'): (95.9, 2.3),
 ('ablation', 'GLI', 'Ours', 4, 'DSC'): (91.3, 8.5),
 ('ablation', 'GLI', 'Ours', 5, 'DSC'): (94.8, 3.9),
 ('ablation', 'GLI', 'Ours', 6, 'DSC'): (98.4, 2.9),
 ('ablation', 'GLI', 'Ours', 7, 'DSC'): (97.2, 3.4),
 ('ablation', 'GLI', 'Ours', 1, 'HD95'): (1.07, 0.29),
 ('ablation', 'GLI', 'Ours', 2, 'HD95'): (1.22, 0.66),
 ('ablation', 'GLI', 'Ours', 3, 'HD95'): (1.1, 0.36),
 ('ablation', 'GLI', 'Ours', 4, 'HD95'): (9.46, 12.46),
 ('ablation', 'GLI', 'Ours', 5, 'HD95'): (1.26, 0.84),
 ('ablation', 'GLI', 'Ours', 6, 'HD95'): (1.21, 1.89),
 ('ablation', 'GLI', 'Ours', 7, 'HD95'): (1.22, 1.7),
 ('ablation', 'WMH', 'Baseline', 1, 'DSC'): (79.3, 2.2),
 ('ablation', 'WMH', 'Baseline', 2, 'DSC'): (88.4, 2.0),
 ('ablation', 'WMH', 'Baseline', 3, 'DSC'): (89.9, 2.8),
 ('ablation', 'WMH', 'Baseline', 4, 'DSC'): (30.4, 27.3),
 ('ablation', 'WMH', 'Baseline', 5, 'DSC'): (89.1, 3.4),
 ('ablation', 'WMH', 'Baseline', 6, 'DSC'): (91.5, 1.2),
 ('ablation', 'WMH', 'Baseline', 7, 'DSC'): (87.5, 2.0),
 ('ablation', 'WMH', 'Baseline', 1, 'HD95'): (6.4, 1.27),
 ('ablation', 'WMH', 'Baseline', 2, 'HD95'): (1.67, 0.33),
 ('ablation', 'WMH', 'Baseline', 3, 'HD95'): (1.63, 0.36),
 ('ablation', 'WMH', 'Baseline', 4, 'HD95'): (135.48, 148.12),
 ('ablation', 'WMH', 'Baseline', 5, 'HD95'): (1.82, 0.36),
 ('ablation', 'WMH', 'Baseline', 6, 'HD95'): (4.86, 0.89),
 ('ablation', 'WMH', 'Baseline', 7, 'HD95'): (2.28, 0.48),
 ('ablation', 'WMH', 'w/o Simplex', 1, 'DSC'): (79.4, 2.1),
 ('ablation', 'WMH', 'w/o Simplex', 2, 'DSC'): (88.4, 2.0),
 ('ablation', 'WMH', 'w/o Simplex', 3, 'DSC'): (90.0, 2.7),
 ('ablation', 'WMH', 'w/o Simplex', 4, 'DSC'): (35.6, 27.0),
 ('ablation', 'WMH', 'w/o Simplex', 5, 'DSC'): (89.1, 3.4),
 ('ablation', 'WMH', 'w/o Simplex', 6, 'DSC'): (91.5, 1.2),
 ('ablation', 'WMH', 'w/o Simplex', 7, 'DSC'): (87.5, 2.0),
 ('ablation', 'WMH', 'w/o Simplex', 1, 'HD95'): (6.4, 1.27),
 ('ablation', 'WMH', 'w/o Simplex', 2, 'HD95'): (1.67, 0.32),
 ('ablation', 'WMH', 'w/o Simplex', 3, 'HD95'): (1.55, 0.3),
 ('ablation', 'WMH', 'w/o Simplex', 4, 'HD95'): (40.19, 50.59),
 ('ablation', 'WMH', 'w/o Simplex', 5, 'HD95'): (1.84, 0.36),
 ('ablation', 'WMH', 'w/o Simplex', 6, 'HD95'): (4.84, 0.87),
 ('ablation', 'WMH', 'w/o Simplex', 7, 'HD95'): (2.28, 0.48),
 ('ablation', 'WMH', 'w/o FDAR', 1, 'DSC'): (79.3, 2.2),
 ('ablation', 'WMH', 'w/o FDAR', 2, 'DSC'): (88.4, 1.9),
 ('ablation', 'WMH', 'w/o FDAR', 3, 'DSC'): (90.1, 2.7),
 ('ablation', 'WMH', 'w/o FDAR', 4, 'DSC'): (34.8, 27.2),
 ('ablation', 'WMH', 'w/o FDAR', 5, 'DSC'): (89.1, 3.4),
 ('ablation', 'WMH', 'w/o FDAR', 6, 'DSC'): (91.5, 1.1),
 ('ablation', 'WMH', 'w/o FDAR', 7, 'DSC'): (87.5, 2.0),
 ('ablation', 'WMH', 'w/o FDAR', 1, 'HD95'): (6.39, 1.27),
 ('ablation', 'WMH', 'w/o FDAR', 2, 'HD95'): (1.67, 0.34),
 ('ablation', 'WMH', 'w/o FDAR', 3, 'HD95'): (1.57, 0.31),
 ('ablation', 'WMH', 'w/o FDAR', 4, 'HD95'): (46.85, 50.77),
 ('ablation', 'WMH', 'w/o FDAR', 5, 'HD95'): (1.84, 0.36),
 ('ablation', 'WMH', 'w/o FDAR', 6, 'HD95'): (4.84, 0.87),
 ('ablation', 'WMH', 'w/o FDAR', 7, 'HD95'): (2.27, 0.47),
 ('ablation', 'WMH', 'w/o MIPB', 1, 'DSC'): (79.3, 2.1),
 ('ablation', 'WMH', 'w/o MIPB', 2, 'DSC'): (88.4, 1.9),
 ('ablation', 'WMH', 'w/o MIPB', 3, 'DSC'): (90.0, 2.8),
 ('ablation', 'WMH', 'w/o MIPB', 4, 'DSC'): (31.5, 27.8),
 ('ablation', 'WMH', 'w/o MIPB', 5, 'DSC'): (89.0, 3.5),
 ('ablation', 'WMH', 'w/o MIPB', 6, 'DSC'): (91.5, 1.1),
 ('ablation', 'WMH', 'w/o MIPB', 7, 'DSC'): (87.6, 2.0),
 ('ablation', 'WMH', 'w/o MIPB', 1, 'HD95'): (6.4, 1.27),
 ('ablation', 'WMH', 'w/o MIPB', 2, 'HD95'): (1.66, 0.32),
 ('ablation', 'WMH', 'w/o MIPB', 3, 'HD95'): (1.61, 0.32),
 ('ablation', 'WMH', 'w/o MIPB', 4, 'HD95'): (119.19, 140.25),
 ('ablation', 'WMH', 'w/o MIPB', 5, 'HD95'): (1.83, 0.36),
 ('ablation', 'WMH', 'w/o MIPB', 6, 'HD95'): (4.85, 0.89),
 ('ablation', 'WMH', 'w/o MIPB', 7, 'HD95'): (2.26, 0.48),
 ('ablation', 'WMH', 'Ours', 1, 'DSC'): (79.4, 2.1),
 ('ablation', 'WMH', 'Ours', 2, 'DSC'): (88.4, 1.9),
 ('ablation', 'WMH', 'Ours', 3, 'DSC'): (90.1, 2.7),
 ('ablation', 'WMH', 'Ours', 4, 'DSC'): (40.1, 25.3),
 ('ablation', 'WMH', 'Ours', 5, 'DSC'): (89.0, 3.4),
 ('ablation', 'WMH', 'Ours', 6, 'DSC'): (91.4, 1.2),
 ('ablation', 'WMH', 'Ours', 7, 'DSC'): (87.5, 2.0),
 ('ablation', 'WMH', 'Ours', 1, 'HD95'): (6.38, 1.26),
 ('ablation', 'WMH', 'Ours', 2, 'HD95'): (1.68, 0.36),
 ('ablation', 'WMH', 'Ours', 3, 'HD95'): (1.5, 0.27),
 ('ablation', 'WMH', 'Ours', 4, 'HD95'): (22.69, 12.18),
 ('ablation', 'WMH', 'Ours', 5, 'HD95'): (1.84, 0.35),
 ('ablation', 'WMH', 'Ours', 6, 'HD95'): (4.85, 0.85),
 ('ablation', 'WMH', 'Ours', 7, 'HD95'): (2.26, 0.43),
 ('ablation', 'MEN', 'Baseline', 1, 'DSC'): (95.8, 1.6),
 ('ablation', 'MEN', 'Baseline', 2, 'DSC'): (96.0, 3.0),
 ('ablation', 'MEN', 'Baseline', 3, 'DSC'): (95.8, 2.0),
 ('ablation', 'MEN', 'Baseline', 4, 'DSC'): (57.6, 37.7),
 ('ablation', 'MEN', 'Baseline', 5, 'DSC'): (95.7, 2.4),
 ('ablation', 'MEN', 'Baseline', 6, 'DSC'): (98.1, 2.0),
 ('ablation', 'MEN', 'Baseline', 7, 'DSC'): (97.3, 1.8),
 ('ablation', 'MEN', 'Baseline', 1, 'HD95'): (1.04, 0.16),
 ('ablation', 'MEN', 'Baseline', 2, 'HD95'): (1.08, 0.31),
 ('ablation', 'MEN', 'Baseline', 3, 'HD95'): (1.09, 0.3),
 ('ablation', 'MEN', 'Baseline', 4, 'HD95'): (95.65, 139.62),
 ('ablation', 'MEN', 'Baseline', 5, 'HD95'): (1.07, 0.24),
 ('ablation', 'MEN', 'Baseline', 6, 'HD95'): (1.27, 0.92),
 ('ablation', 'MEN', 'Baseline', 7, 'HD95'): (1.1, 0.33),
 ('ablation', 'MEN', 'w/o Simplex', 1, 'DSC'): (95.8, 1.6),
 ('ablation', 'MEN', 'w/o Simplex', 2, 'DSC'): (96.0, 3.0),
 ('ablation', 'MEN', 'w/o Simplex', 3, 'DSC'): (95.8, 2.0),
 ('ablation', 'MEN', 'w/o Simplex', 4, 'DSC'): (58.8, 36.6),
 ('ablation', 'MEN', 'w/o Simplex', 5, 'DSC'): (95.7, 2.4),
 ('ablation', 'MEN', 'w/o Simplex', 6, 'DSC'): (98.1, 1.8),
 ('ablation', 'MEN', 'w/o Simplex', 7, 'DSC'): (97.3, 1.8),
 ('ablation', 'MEN', 'w/o Simplex', 1, 'HD95'): (1.04, 0.16),
 ('ablation', 'MEN', 'w/o Simplex', 2, 'HD95'): (1.08, 0.31),
 ('ablation', 'MEN', 'w/o Simplex', 3, 'HD95'): (1.09, 0.28),
 ('ablation', 'MEN', 'w/o Simplex', 4, 'HD95'): (49.9, 86.53),
 ('ablation', 'MEN', 'w/o Simplex', 5, 'HD95'): (1.07, 0.25),
 ('ablation', 'MEN', 'w/o Simplex', 6, 'HD95'): (1.24, 0.82),
 ('ablation', 'MEN', 'w/o Simplex', 7, 'HD95'): (1.09, 0.36),
 ('ablation', 'MEN', 'w/o FDAR', 1, 'DSC'): (95.8, 1.6),
 ('ablation', 'MEN', 'w/o FDAR', 2, 'DSC'): (96.0, 3.0),
 ('ablation', 'MEN', 'w/o FDAR', 3, 'DSC'): (95.8, 2.0),
 ('ablation', 'MEN', 'w/o FDAR', 4, 'DSC'): (59.0, 36.3),
 ('ablation', 'MEN', 'w/o FDAR', 5, 'DSC'): (95.7, 2.4),
 ('ablation', 'MEN', 'w/o FDAR', 6, 'DSC'): (98.1, 1.9),
 ('ablation', 'MEN', 'w/o FDAR', 7, 'DSC'): (97.3, 1.8),
 ('ablation', 'MEN', 'w/o FDAR', 1, 'HD95'): (1.04, 0.16),
 ('ablation', 'MEN', 'w/o FDAR', 2, 'HD95'): (1.08, 0.31),
 ('ablation', 'MEN', 'w/o FDAR', 3, 'HD95'): (1.09, 0.28),
 ('ablation', 'MEN', 'w/o FDAR', 4, 'HD95'): (58.77, 97.69),
 ('ablation', 'MEN', 'w/o FDAR', 5, 'HD95'): (1.07, 0.24),
 ('ablation', 'MEN', 'w/o FDAR', 6, 'HD95'): (1.25, 0.87),
 ('ablation', 'MEN', 'w/o FDAR', 7, 'HD95'): (1.09, 0.36),
 ('ablation', 'MEN', 'w/o MIPB', 1, 'DSC'): (95.8, 1.6),
 ('ablation', 'MEN', 'w/o MIPB', 2, 'DSC'): (96.0, 3.0),
 ('ablation', 'MEN', 'w/o MIPB', 3, 'DSC'): (95.8, 2.0),
 ('ablation', 'MEN', 'w/o MIPB', 4, 'DSC'): (57.5, 37.6),
 ('ablation', 'MEN', 'w/o MIPB', 5, 'DSC'): (95.7, 2.4),
 ('ablation', 'MEN', 'w/o MIPB', 6, 'DSC'): (98.1, 1.9),
 ('ablation', 'MEN', 'w/o MIPB', 7, 'DSC'): (97.2, 1.8),
 ('ablation', 'MEN', 'w/o MIPB', 1, 'HD95'): (1.04, 0.17),
 ('ablation', 'MEN', 'w/o MIPB', 2, 'HD95'): (1.08, 0.31),
 ('ablation', 'MEN', 'w/o MIPB', 3, 'HD95'): (1.09, 0.28),
 ('ablation', 'MEN', 'w/o MIPB', 4, 'HD95'): (93.22, 136.75),
 ('ablation', 'MEN', 'w/o MIPB', 5, 'HD95'): (1.07, 0.27),
 ('ablation', 'MEN', 'w/o MIPB', 6, 'HD95'): (1.28, 1.03),
 ('ablation', 'MEN', 'w/o MIPB', 7, 'HD95'): (1.09, 0.37),
 ('ablation', 'MEN', 'Ours', 1, 'DSC'): (95.8, 1.6),
 ('ablation', 'MEN', 'Ours', 2, 'DSC'): (96.0, 3.0),
 ('ablation', 'MEN', 'Ours', 3, 'DSC'): (95.8, 2.0),
 ('ablation', 'MEN', 'Ours', 4, 'DSC'): (58.9, 36.1),
 ('ablation', 'MEN', 'Ours', 5, 'DSC'): (95.7, 2.4),
 ('ablation', 'MEN', 'Ours', 6, 'DSC'): (98.1, 1.8),
 ('ablation', 'MEN', 'Ours', 7, 'DSC'): (97.3, 1.8),
 ('ablation', 'MEN', 'Ours', 1, 'HD95'): (1.04, 0.15),
 ('ablation', 'MEN', 'Ours', 2, 'HD95'): (1.08, 0.31),
 ('ablation', 'MEN', 'Ours', 3, 'HD95'): (1.09, 0.27),
 ('ablation', 'MEN', 'Ours', 4, 'HD95'): (42.22, 73.6),
 ('ablation', 'MEN', 'Ours', 5, 'HD95'): (1.07, 0.25),
 ('ablation', 'MEN', 'Ours', 6, 'HD95'): (1.24, 0.81),
 ('ablation', 'MEN', 'Ours', 7, 'HD95'): (1.1, 0.37),
 ('ablation', 'MRBrainS18', 'Baseline', 1, 'DSC'): (71.9, 2.4),
 ('ablation', 'MRBrainS18', 'Baseline', 2, 'DSC'): (65.0, 2.7),
 ('ablation', 'MRBrainS18', 'Baseline', 3, 'DSC'): (80.0, 3.3),
 ('ablation', 'MRBrainS18', 'Baseline', 4, 'DSC'): (20.9, 27.8),
 ('ablation', 'MRBrainS18', 'Baseline', 5, 'DSC'): (86.1, 3.6),
 ('ablation', 'MRBrainS18', 'Baseline', 6, 'DSC'): (88.7, 2.5),
 ('ablation', 'MRBrainS18', 'Baseline', 7, 'DSC'): (75.2, 3.3),
 ('ablation', 'MRBrainS18', 'Baseline', 1, 'HD95'): (9.38, 1.59),
 ('ablation', 'MRBrainS18', 'Baseline', 2, 'HD95'): (12.26, 1.41),
 ('ablation', 'MRBrainS18', 'Baseline', 3, 'HD95'): (2.51, 0.35),
 ('ablation', 'MRBrainS18', 'Baseline', 4, 'HD95'): (180.73, 161.74),
 ('ablation', 'MRBrainS18', 'Baseline', 5, 'HD95'): (3.36, 1.1),
 ('ablation', 'MRBrainS18', 'Baseline', 6, 'HD95'): (6.01, 1.19),
 ('ablation', 'MRBrainS18', 'Baseline', 7, 'HD95'): (7.37, 1.34),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 1, 'DSC'): (72.0, 2.3),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 2, 'DSC'): (65.1, 2.7),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 3, 'DSC'): (80.1, 3.3),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 4, 'DSC'): (21.2, 24.2),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 5, 'DSC'): (86.1, 3.6),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 6, 'DSC'): (88.7, 2.5),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 7, 'DSC'): (75.3, 3.3),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 1, 'HD95'): (9.35, 1.58),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 2, 'HD95'): (12.27, 1.4),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 3, 'HD95'): (2.42, 0.29),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 4, 'HD95'): (65.93, 87.48),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 5, 'HD95'): (3.41, 1.1),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 6, 'HD95'): (6.02, 1.19),
 ('ablation', 'MRBrainS18', 'w/o Simplex', 7, 'HD95'): (7.36, 1.35),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 1, 'DSC'): (72.0, 2.4),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 2, 'DSC'): (65.1, 2.7),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 3, 'DSC'): (80.1, 3.3),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 4, 'DSC'): (21.4, 23.8),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 5, 'DSC'): (86.1, 3.5),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 6, 'DSC'): (88.7, 2.5),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 7, 'DSC'): (75.3, 3.3),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 1, 'HD95'): (9.34, 1.6),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 2, 'HD95'): (12.26, 1.4),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 3, 'HD95'): (2.45, 0.29),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 4, 'HD95'): (86.47, 116.55),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 5, 'HD95'): (3.41, 1.12),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 6, 'HD95'): (6.01, 1.18),
 ('ablation', 'MRBrainS18', 'w/o FDAR', 7, 'HD95'): (7.36, 1.33),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 1, 'DSC'): (72.1, 2.3),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 2, 'DSC'): (65.0, 2.7),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 3, 'DSC'): (80.1, 3.3),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 4, 'DSC'): (21.9, 28.3),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 5, 'DSC'): (86.1, 3.6),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 6, 'DSC'): (88.7, 2.5),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 7, 'DSC'): (75.3, 3.3),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 1, 'HD95'): (9.33, 1.61),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 2, 'HD95'): (12.25, 1.4),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 3, 'HD95'): (2.5, 0.34),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 4, 'HD95'): (150.67, 150.63),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 5, 'HD95'): (3.38, 1.11),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 6, 'HD95'): (6.01, 1.18),
 ('ablation', 'MRBrainS18', 'w/o MIPB', 7, 'HD95'): (7.35, 1.36),
 ('ablation', 'MRBrainS18', 'Ours', 1, 'DSC'): (72.1, 2.3),
 ('ablation', 'MRBrainS18', 'Ours', 2, 'DSC'): (65.0, 2.7),
 ('ablation', 'MRBrainS18', 'Ours', 3, 'DSC'): (80.2, 3.2),
 ('ablation', 'MRBrainS18', 'Ours', 4, 'DSC'): (25.4, 23.3),
 ('ablation', 'MRBrainS18', 'Ours', 5, 'DSC'): (86.1, 3.6),
 ('ablation', 'MRBrainS18', 'Ours', 6, 'DSC'): (88.7, 2.5),
 ('ablation', 'MRBrainS18', 'Ours', 7, 'DSC'): (75.2, 3.3),
 ('ablation', 'MRBrainS18', 'Ours', 1, 'HD95'): (9.34, 1.58),
 ('ablation', 'MRBrainS18', 'Ours', 2, 'HD95'): (12.26, 1.4),
 ('ablation', 'MRBrainS18', 'Ours', 3, 'HD95'): (2.37, 0.24),
 ('ablation', 'MRBrainS18', 'Ours', 4, 'HD95'): (40.49, 64.03),
 ('ablation', 'MRBrainS18', 'Ours', 5, 'HD95'): (3.41, 1.1),
 ('ablation', 'MRBrainS18', 'Ours', 6, 'HD95'): (6.01, 1.19),
 ('ablation', 'MRBrainS18', 'Ours', 7, 'HD95'): (7.36, 1.35),
 ('ablation', 'Stroke', 'Baseline', 4, 'DSC'): (46.1, 26.2),
 ('ablation', 'Stroke', 'w/o Simplex', 4, 'DSC'): (49.0, 25.3),
 ('ablation', 'Stroke', 'w/o FDAR', 4, 'DSC'): (49.8, 24.9),
 ('ablation', 'Stroke', 'w/o MIPB', 4, 'DSC'): (46.7, 25.9),
 ('ablation', 'Stroke', 'Ours', 4, 'DSC'): (50.6, 24.5),
 ('ablation', 'Stroke', 'Baseline', 4, 'HD95'): (58.37, 84.94),
 ('ablation', 'Stroke', 'w/o Simplex', 4, 'HD95'): (26.84, 29.11),
 ('ablation', 'Stroke', 'w/o FDAR', 4, 'HD95'): (40.36, 65.42),
 ('ablation', 'Stroke', 'w/o MIPB', 4, 'HD95'): (54.22, 79.26),
 ('ablation', 'Stroke', 'Ours', 4, 'HD95'): (18.9, 10.81),
 ('ablation', 'MET', 'Baseline', 4, 'DSC'): (64.4, 33.3),
 ('ablation', 'MET', 'w/o Simplex', 4, 'DSC'): (65.8, 32.0),
 ('ablation', 'MET', 'w/o FDAR', 4, 'DSC'): (66.0, 32.0),
 ('ablation', 'MET', 'w/o MIPB', 4, 'DSC'): (65.1, 32.7),
 ('ablation', 'MET', 'Ours', 4, 'DSC'): (65.9, 31.7),
 ('ablation', 'MET', 'Baseline', 4, 'HD95'): (68.64, 109.35),
 ('ablation', 'MET', 'w/o Simplex', 4, 'HD95'): (42.66, 75.33),
 ('ablation', 'MET', 'w/o FDAR', 4, 'HD95'): (44.43, 74.59),
 ('ablation', 'MET', 'w/o MIPB', 4, 'HD95'): (68.38, 109.35),
 ('ablation', 'MET', 'Ours', 4, 'HD95'): (35.4, 56.93),
 ('ablation', 'MS', 'Baseline', 4, 'DSC'): (25.8, 21.9),
 ('ablation', 'MS', 'w/o Simplex', 4, 'DSC'): (34.8, 18.8),
 ('ablation', 'MS', 'w/o FDAR', 4, 'DSC'): (34.0, 17.4),
 ('ablation', 'MS', 'w/o MIPB', 4, 'DSC'): (28.3, 21.1),
 ('ablation', 'MS', 'Ours', 4, 'DSC'): (40.4, 16.8),
 ('ablation', 'MS', 'Baseline', 4, 'HD95'): (102.49, 111.43),
 ('ablation', 'MS', 'w/o Simplex', 4, 'HD95'): (42.76, 21.97),
 ('ablation', 'MS', 'w/o FDAR', 4, 'HD95'): (45.24, 18.61),
 ('ablation', 'MS', 'w/o MIPB', 4, 'HD95'): (69.1, 70.05),
 ('ablation', 'MS', 'Ours', 4, 'HD95'): (31.24, 14.71),
 ('ablation', 'PED', 'Baseline', 4, 'DSC'): (90.1, 3.8),
 ('ablation', 'PED', 'w/o Simplex', 4, 'DSC'): (89.9, 4.0),
 ('ablation', 'PED', 'w/o FDAR', 4, 'DSC'): (90.1, 3.8),
 ('ablation', 'PED', 'w/o MIPB', 4, 'DSC'): (89.9, 4.4),
 ('ablation', 'PED', 'Ours', 4, 'DSC'): (90.2, 3.8),
 ('ablation', 'PED', 'Baseline', 4, 'HD95'): (6.45, 10.69),
 ('ablation', 'PED', 'w/o Simplex', 4, 'HD95'): (6.47, 10.64),
 ('ablation', 'PED', 'w/o FDAR', 4, 'HD95'): (6.46, 10.64),
 ('ablation', 'PED', 'w/o MIPB', 4, 'HD95'): (6.55, 10.71),
 ('ablation', 'PED', 'Ours', 4, 'HD95'): (6.09, 10.61),
 ('ablation', 'SSA', 'Baseline', 4, 'DSC'): (93.2, 2.4),
 ('ablation', 'SSA', 'w/o Simplex', 4, 'DSC'): (93.2, 2.6),
 ('ablation', 'SSA', 'w/o FDAR', 4, 'DSC'): (93.2, 2.5),
 ('ablation', 'SSA', 'w/o MIPB', 4, 'DSC'): (93.5, 2.0),
 ('ablation', 'SSA', 'Ours', 4, 'DSC'): (93.3, 2.1),
 ('ablation', 'SSA', 'Baseline', 4, 'HD95'): (7.03, 6.11),
 ('ablation', 'SSA', 'w/o Simplex', 4, 'HD95'): (7.54, 7.18),
 ('ablation', 'SSA', 'w/o FDAR', 4, 'HD95'): (6.19, 5.28),
 ('ablation', 'SSA', 'w/o MIPB', 4, 'HD95'): (6.48, 5.08),
 ('ablation', 'SSA', 'Ours', 4, 'HD95'): (8.93, 11.88),
 ('wmh_source', 'GLI', 'WMH-Baseline', 1, 'DSC'): (79.5, 4.2),
 ('wmh_source', 'GLI', 'WMH-Baseline', 2, 'DSC'): (74.0, 14.5),
 ('wmh_source', 'GLI', 'WMH-Baseline', 3, 'DSC'): (86.2, 6.3),
 ('wmh_source', 'GLI', 'WMH-Baseline', 4, 'DSC'): (69.8, 23.7),
 ('wmh_source', 'GLI', 'WMH-Baseline', 5, 'DSC'): (76.6, 15.4),
 ('wmh_source', 'GLI', 'WMH-Baseline', 6, 'DSC'): (77.9, 22.8),
 ('wmh_source', 'GLI', 'WMH-Baseline', 7, 'DSC'): (68.3, 25.2),
 ('wmh_source', 'GLI', 'WMH-Baseline', 1, 'HD95'): (8.31, 3.24),
 ('wmh_source', 'GLI', 'WMH-Baseline', 2, 'HD95'): (9.26, 9.82),
 ('wmh_source', 'GLI', 'WMH-Baseline', 3, 'HD95'): (3.83, 3.45),
 ('wmh_source', 'GLI', 'WMH-Baseline', 4, 'HD95'): (11.32, 9.08),
 ('wmh_source', 'GLI', 'WMH-Baseline', 5, 'HD95'): (11.87, 14.48),
 ('wmh_source', 'GLI', 'WMH-Baseline', 6, 'HD95'): (16.32, 22.33),
 ('wmh_source', 'GLI', 'WMH-Baseline', 7, 'HD95'): (12.48, 16.88),
 ('wmh_source', 'GLI', 'WMH-Ours', 1, 'DSC'): (80.5, 3.6),
 ('wmh_source', 'GLI', 'WMH-Ours', 2, 'DSC'): (77.7, 11.9),
 ('wmh_source', 'GLI', 'WMH-Ours', 3, 'DSC'): (87.1, 5.7),
 ('wmh_source', 'GLI', 'WMH-Ours', 4, 'DSC'): (82.5, 11.6),
 ('wmh_source', 'GLI', 'WMH-Ours', 5, 'DSC'): (80.2, 11.0),
 ('wmh_source', 'GLI', 'WMH-Ours', 6, 'DSC'): (83.2, 17.6),
 ('wmh_source', 'GLI', 'WMH-Ours', 7, 'DSC'): (71.3, 24.1),
 ('wmh_source', 'GLI', 'WMH-Ours', 1, 'HD95'): (7.62, 2.92),
 ('wmh_source', 'GLI', 'WMH-Ours', 2, 'HD95'): (6.46, 8.64),
 ('wmh_source', 'GLI', 'WMH-Ours', 3, 'HD95'): (3.07, 2.66),
 ('wmh_source', 'GLI', 'WMH-Ours', 4, 'HD95'): (12.97, 12.99),
 ('wmh_source', 'GLI', 'WMH-Ours', 5, 'HD95'): (6.05, 8.69),
 ('wmh_source', 'GLI', 'WMH-Ours', 6, 'HD95'): (11.66, 27.55),
 ('wmh_source', 'GLI', 'WMH-Ours', 7, 'HD95'): (12.39, 27.97),
 ('wmh_source', 'WMH', 'WMH-Baseline', 1, 'DSC'): (95.0, 1.1),
 ('wmh_source', 'WMH', 'WMH-Baseline', 2, 'DSC'): (95.7, 1.2),
 ('wmh_source', 'WMH', 'WMH-Baseline', 3, 'DSC'): (95.8, 2.0),
 ('wmh_source', 'WMH', 'WMH-Baseline', 4, 'DSC'): (76.0, 10.9),
 ('wmh_source', 'WMH', 'WMH-Baseline', 5, 'DSC'): (97.2, 1.2),
 ('wmh_source', 'WMH', 'WMH-Baseline', 6, 'DSC'): (98.1, 0.3),
 ('wmh_source', 'WMH', 'WMH-Baseline', 7, 'DSC'): (96.9, 0.9),
 ('wmh_source', 'WMH', 'WMH-Baseline', 1, 'HD95'): (1.21, 0.25),
 ('wmh_source', 'WMH', 'WMH-Baseline', 2, 'HD95'): (1.02, 0.12),
 ('wmh_source', 'WMH', 'WMH-Baseline', 3, 'HD95'): (1.02, 0.13),
 ('wmh_source', 'WMH', 'WMH-Baseline', 4, 'HD95'): (4.03, 5.65),
 ('wmh_source', 'WMH', 'WMH-Baseline', 5, 'HD95'): (1.01, 0.06),
 ('wmh_source', 'WMH', 'WMH-Baseline', 6, 'HD95'): (1.0, 0.0),
 ('wmh_source', 'WMH', 'WMH-Baseline', 7, 'HD95'): (1.04, 0.19),
 ('wmh_source', 'WMH', 'WMH-Ours', 1, 'DSC'): (95.0, 1.1),
 ('wmh_source', 'WMH', 'WMH-Ours', 2, 'DSC'): (95.8, 1.2),
 ('wmh_source', 'WMH', 'WMH-Ours', 3, 'DSC'): (95.8, 2.0),
 ('wmh_source', 'WMH', 'WMH-Ours', 4, 'DSC'): (75.7, 11.3),
 ('wmh_source', 'WMH', 'WMH-Ours', 5, 'DSC'): (97.3, 1.2),
 ('wmh_source', 'WMH', 'WMH-Ours', 6, 'DSC'): (98.2, 0.3),
 ('wmh_source', 'WMH', 'WMH-Ours', 7, 'DSC'): (97.0, 0.9),
 ('wmh_source', 'WMH', 'WMH-Ours', 1, 'HD95'): (1.21, 0.25),
 ('wmh_source', 'WMH', 'WMH-Ours', 2, 'HD95'): (1.02, 0.14),
 ('wmh_source', 'WMH', 'WMH-Ours', 3, 'HD95'): (1.02, 0.12),
 ('wmh_source', 'WMH', 'WMH-Ours', 4, 'HD95'): (4.38, 5.88),
 ('wmh_source', 'WMH', 'WMH-Ours', 5, 'HD95'): (1.01, 0.06),
 ('wmh_source', 'WMH', 'WMH-Ours', 6, 'HD95'): (1.0, 0.0),
 ('wmh_source', 'WMH', 'WMH-Ours', 7, 'HD95'): (1.03, 0.15),
 ('wmh_source', 'MEN', 'WMH-Baseline', 1, 'DSC'): (81.3, 3.6),
 ('wmh_source', 'MEN', 'WMH-Baseline', 2, 'DSC'): (77.9, 14.8),
 ('wmh_source', 'MEN', 'WMH-Baseline', 3, 'DSC'): (87.0, 6.0),
 ('wmh_source', 'MEN', 'WMH-Baseline', 4, 'DSC'): (45.3, 30.2),
 ('wmh_source', 'MEN', 'WMH-Baseline', 5, 'DSC'): (78.3, 13.1),
 ('wmh_source', 'MEN', 'WMH-Baseline', 6, 'DSC'): (82.7, 19.7),
 ('wmh_source', 'MEN', 'WMH-Baseline', 7, 'DSC'): (78.2, 20.4),
 ('wmh_source', 'MEN', 'WMH-Baseline', 1, 'HD95'): (7.13, 3.29),
 ('wmh_source', 'MEN', 'WMH-Baseline', 2, 'HD95'): (7.67, 9.97),
 ('wmh_source', 'MEN', 'WMH-Baseline', 3, 'HD95'): (2.99, 3.06),
 ('wmh_source', 'MEN', 'WMH-Baseline', 4, 'HD95'): (26.89, 22.42),
 ('wmh_source', 'MEN', 'WMH-Baseline', 5, 'HD95'): (7.66, 13.18),
 ('wmh_source', 'MEN', 'WMH-Baseline', 6, 'HD95'): (16.05, 24.63),
 ('wmh_source', 'MEN', 'WMH-Baseline', 7, 'HD95'): (11.4, 30.26),
 ('wmh_source', 'MEN', 'WMH-Ours', 1, 'DSC'): (81.9, 3.3),
 ('wmh_source', 'MEN', 'WMH-Ours', 2, 'DSC'): (80.3, 12.8),
 ('wmh_source', 'MEN', 'WMH-Ours', 3, 'DSC'): (87.5, 5.6),
 ('wmh_source', 'MEN', 'WMH-Ours', 4, 'DSC'): (59.2, 29.8),
 ('wmh_source', 'MEN', 'WMH-Ours', 5, 'DSC'): (80.4, 9.0),
 ('wmh_source', 'MEN', 'WMH-Ours', 6, 'DSC'): (85.9, 16.8),
 ('wmh_source', 'MEN', 'WMH-Ours', 7, 'DSC'): (80.2, 19.8),
 ('wmh_source', 'MEN', 'WMH-Ours', 1, 'HD95'): (6.59, 3.03),
 ('wmh_source', 'MEN', 'WMH-Ours', 2, 'HD95'): (5.03, 7.07),
 ('wmh_source', 'MEN', 'WMH-Ours', 3, 'HD95'): (2.65, 2.51),
 ('wmh_source', 'MEN', 'WMH-Ours', 4, 'HD95'): (29.39, 25.35),
 ('wmh_source', 'MEN', 'WMH-Ours', 5, 'HD95'): (4.29, 7.7),
 ('wmh_source', 'MEN', 'WMH-Ours', 6, 'HD95'): (10.27, 18.57),
 ('wmh_source', 'MEN', 'WMH-Ours', 7, 'HD95'): (7.55, 16.03),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 1, 'DSC'): (83.4, 1.3),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 2, 'DSC'): (67.6, 2.2),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 3, 'DSC'): (82.1, 2.8),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 4, 'DSC'): (57.4, 23.6),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 5, 'DSC'): (86.4, 3.4),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 6, 'DSC'): (91.2, 1.4),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 7, 'DSC'): (78.0, 3.9),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 1, 'HD95'): (2.98, 0.49),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 2, 'HD95'): (11.86, 1.35),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 3, 'HD95'): (1.89, 0.21),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 4, 'HD95'): (18.75, 67.68),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 5, 'HD95'): (4.23, 2.07),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 6, 'HD95'): (3.23, 0.68),
 ('wmh_source', 'MRBrainS18', 'WMH-Baseline', 7, 'HD95'): (7.41, 1.44),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 1, 'DSC'): (83.4, 1.3),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 2, 'DSC'): (67.6, 2.2),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 3, 'DSC'): (82.1, 2.8),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 4, 'DSC'): (57.5, 23.4),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 5, 'DSC'): (86.4, 3.4),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 6, 'DSC'): (91.2, 1.4),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 7, 'DSC'): (78.0, 3.9),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 1, 'HD95'): (3.01, 0.49),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 2, 'HD95'): (11.87, 1.35),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 3, 'HD95'): (1.9, 0.22),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 4, 'HD95'): (19.11, 67.63),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 5, 'HD95'): (4.16, 1.96),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 6, 'HD95'): (3.22, 0.68),
 ('wmh_source', 'MRBrainS18', 'WMH-Ours', 7, 'HD95'): (7.41, 1.44),
 ('wmh_source', 'Stroke', 'WMH-Baseline', 4, 'DSC'): (49.3, 29.4),
 ('wmh_source', 'MET', 'WMH-Baseline', 4, 'DSC'): (62.5, 26.9),
 ('wmh_source', 'MS', 'WMH-Baseline', 4, 'DSC'): (61.0, 12.2),
 ('wmh_source', 'PED', 'WMH-Baseline', 4, 'DSC'): (51.2, 35.5),
 ('wmh_source', 'SSA', 'WMH-Baseline', 4, 'DSC'): (64.8, 20.7),
 ('wmh_source', 'Stroke', 'WMH-Baseline', 4, 'HD95'): (17.33, 14.61),
 ('wmh_source', 'MET', 'WMH-Baseline', 4, 'HD95'): (14.81, 16.49),
 ('wmh_source', 'MS', 'WMH-Baseline', 4, 'HD95'): (19.76, 14.18),
 ('wmh_source', 'PED', 'WMH-Baseline', 4, 'HD95'): (22.87, 20.62),
 ('wmh_source', 'SSA', 'WMH-Baseline', 4, 'HD95'): (13.41, 7.72),
 ('wmh_source', 'Stroke', 'WMH-Ours', 4, 'DSC'): (57.5, 24.2),
 ('wmh_source', 'MET', 'WMH-Ours', 4, 'DSC'): (68.6, 22.2),
 ('wmh_source', 'MS', 'WMH-Ours', 4, 'DSC'): (55.6, 15.3),
 ('wmh_source', 'PED', 'WMH-Ours', 4, 'DSC'): (75.5, 19.1),
 ('wmh_source', 'SSA', 'WMH-Ours', 4, 'DSC'): (85.1, 7.0),
 ('wmh_source', 'Stroke', 'WMH-Ours', 4, 'HD95'): (13.99, 12.84),
 ('wmh_source', 'MET', 'WMH-Ours', 4, 'HD95'): (18.84, 19.68),
 ('wmh_source', 'MS', 'WMH-Ours', 4, 'HD95'): (21.88, 14.91),
 ('wmh_source', 'PED', 'WMH-Ours', 4, 'HD95'): (30.13, 27.09),
 ('wmh_source', 'SSA', 'WMH-Ours', 4, 'HD95'): (15.03, 13.77)}

BASELINE_GEOMETRY = {('main', 'MRBrainS18'): {'size': (8.6, 7.8),
                          'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('main', 'WMH'): {'size': (8.6, 7.8),
                   'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('ablation', None): {'size': (10.75, 9.75),
                      'axis_pixels': (213.60000000000005, 93.60000000000001, 2152.8, 2152.8)},
 ('ablation', 'MRBrainS18'): {'size': (8.6, 7.8),
                              'axis_pixels': (170.88000000000005,
                                              74.88,
                                              1722.2400000000002,
                                              1722.2400000000002)},
 ('ablation', 'WMH'): {'size': (8.6, 7.8),
                       'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('wmh_source', 'MRBrainS18'): {'size': (8.6, 7.8),
                                'axis_pixels': (170.88000000000005,
                                                74.88,
                                                1722.2400000000002,
                                                1722.2400000000002)},
 ('main', None): {'size': (10.75, 9.75), 'axis_pixels': (213.60000000000005, 93.60000000000001, 2152.8, 2152.8)},
 ('main', 'GLI'): {'size': (8.6, 7.8),
                   'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('main', 'MEN'): {'size': (8.6, 7.8),
                   'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('ablation', 'GLI'): {'size': (8.6, 7.8),
                       'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('ablation', 'MEN'): {'size': (8.6, 7.8),
                       'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('wmh_source', None): {'size': (10.75, 9.75), 'axis_pixels': (213.60000000000005, 93.60000000000001, 2152.8, 2152.8)},
 ('wmh_source', 'GLI'): {'size': (8.6, 7.8),
                         'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('wmh_source', 'WMH'): {'size': (8.6, 7.8),
                         'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)},
 ('wmh_source', 'MEN'): {'size': (8.6, 7.8),
                         'axis_pixels': (170.88000000000005, 74.88, 1722.2400000000002, 1722.2400000000002)}}


plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
METRIC_LABELS = ['DSC (%) ↑', 'HD95 (mm) ↓']
FROZEN_TICKS = {}
STATS = {'figures': 0, 'closed_curves': 0, 'ordinary_points': 0, 'partial_points': 0,
         'upper_labels': 0, 'integer_ticks': 0, 'decimal_ticks': 0, 'expanded_axes': 0,
         'target_axes': 0, 'source_cells': set()}

class Scale:
    def __init__(self, config, factor=1):
        origin, self.end, self.decimals = config
        self.original_origin = origin
        self.factor = factor
        self.origin = self.end + factor * (origin - self.end) if factor != 1 else origin
        self.ticks = [self.value(r) for r in [.25, .5, .75, 1]]

    def radius(self, value):
        return (value - self.origin) / (self.end - self.origin)

    def value(self, radius):
        return self.origin + radius * (self.end - self.origin)

    def label(self, value):
        return f'{value:.{self.decimals}f}'


def build_scales(family, dataset, axes, methods, partial, expand):
    scales = []
    ours = 'WMH-Ours' if family == 'wmh_source' else 'Ours'
    for item, metric, _ in axes:
        key = family, dataset, item[2], metric
        selected = expand and item[2] in ADJUST_AXES.get((family, dataset, metric), [])
        old = Scale(SCALE_CONFIG[key])
        new = Scale(SCALE_CONFIG[key], 2 if selected else 1)
        values = np.array([DATA[sourcekey(family, item, m, metric)][0]
                           for m in methods + partial if sourcekey(family, item, m, metric) in DATA])
        before = np.array([old.radius(v) for v in values])
        after = np.array([new.radius(v) for v in values])
        if selected:
            ov = DATA[sourcekey(family, item, ours, metric)][0]
            assert ov < max(values) if metric == 'DSC' else ov > min(values), key
            assert np.isclose(abs(new.end-new.origin), 2*abs(old.end-old.origin)), key
            assert np.allclose(after, 1+(before-1)/2, atol=1e-12), key
            assert np.allclose(after[:,None]-after[None,:], (before[:,None]-before[None,:])/2, atol=1e-12), key
            print(f'  AXIS {family}/{dataset or "lesions"}/{item[2]}/{metric}: '
                  f'origin {old.origin:g} -> {new.origin:g}; end {new.end:g}', flush=True)
        else:
            assert np.array_equal(before, after), key
        assert np.array_equal(np.argsort(before, kind='stable'), np.argsort(after, kind='stable')), key
        assert np.all(after >= 0) and np.all(after <= 1.055), key
        assert all(abs(t-round(t)) < 1e-7 for t in new.ticks) if metric == 'DSC' else True
        scales.append(new)
    return scales


def sourcekey(family, item, method, metric):
    ds, label, _ = item
    return family, ds, method, label, metric

def is_reference(family, ds, method, label):
    return family == 'main' and label == 4 and ds in ['GLI','PED','SSA'] and method in ['Lesion-Only','Joint-Baseline']

def text_box(text, renderer):
    patch = text.get_bbox_patch()
    return patch.get_window_extent(renderer) if patch else text.get_window_extent(renderer)

def category_labels(labels):
    return [t for t in labels if t.get_bbox_patch() and t.get_text() not in METRIC_LABELS]

def set_fixed_geometry(fig, ax, job, baseline):
    width, height = baseline[job]['size']
    # Keep the original axes square and radar diameter; add only exterior room
    # for the longer labels, which is trimmed to the actual content on export.
    ax_rect = baseline[job]['axis_pixels']
    margin = 0.7
    fig.set_size_inches(width + 2 * margin, height + 0.8)
    pxwidth, pxheight = fig.get_size_inches() * fig.dpi
    left, bottom, axwidth, axheight = ax_rect
    ax.set_position([(left + margin * fig.dpi) / pxwidth,
                     (bottom + 0.2 * fig.dpi) / pxheight,
                     axwidth / pxwidth, axheight / pxheight])

def position_legend(fig, ax, legend, labels, renderer):
    # An annotation is attached immediately after its own dataset label.
    # Use the rendered box center, including when the label itself is left- or
    # right-aligned, and anchor the UB glyph box four points beyond its edge.
    pp = fig.dpi / 72
    for i, text in enumerate(labels):
        if not text.get_text().startswith('UB '):
            continue
        name = labels[i - 1]
        b = text_box(name, renderer)
        upper = name.get_position()[1] > 0
        text.xy = ax.transData.inverted().transform(((b.x0 + b.x1) / 2, b.y1 if upper else b.y0))
        text.set_ha('center')
        text.set_va('bottom' if upper else 'top')
        text.set_position((0, 4 if upper else -4))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text_box(t, renderer) for t in labels]
    right = max(text_box(t, renderer).x1 for t in category_labels(labels))
    circle_top = ax.transData.transform((0, 1.055))[1]
    top = max(max(b.y1 for b in boxes), circle_top + legend.get_window_extent(renderer).height + 4 * pp)
    for extra in range(0, 161, 2):
        legend.set_bbox_to_anchor((right / fig.bbox.width, (top + extra) / fig.bbox.height), transform=fig.transFigure)
        lb = legend.get_window_extent(renderer)
        if all(not lb.overlaps(b.padded(4 * pp)) for b in boxes):
            assert lb.y1 < fig.bbox.height - 5, 'Legend requires more top room'
            break
    else:
        raise AssertionError('Legend cannot clear top labels')

    metrics = []
    for label, metric in zip(METRIC_LABELS, ['DSC', 'HD95']):
        face, edge = METRIC_COLORS[metric]
        text = ax.text(0, 0, label, fontsize=14, ha='right', va='center',
                       color='#161616', zorder=10, clip_on=False,
                       bbox=dict(boxstyle='round,pad=.24,rounding_size=.22',
                                 fc=face, ec=edge, lw=.8))
        metrics.append(text)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [text_box(t, renderer).width for t in metrics]
    heights = [text_box(t, renderer).height for t in metrics]
    origin = ax.transData.transform((0, 0))
    radius = ax.transData.transform((1.055, 0))[0] - origin[0]
    bottom = min(b.y0 for b in boxes)
    from matplotlib.transforms import Bbox
    # Stack both boxes in the free lower-right corner. Prefer the current
    # category envelope, moving only downward if that corner is occupied.
    for extra in range(0, 241, 2):
        y0 = bottom - extra
        proposed = [Bbox.from_bounds(right - widths[0], y0 + heights[1] + 6 * pp, widths[0], heights[0]),
                    Bbox.from_bounds(right - widths[1], y0, widths[1], heights[1])]
        if any(a.overlaps(b.padded(4 * pp)) for a in proposed for b in boxes):
            continue
        clear = True
        for a in proposed:
            dx = max(a.x0 - origin[0], origin[0] - a.x1, 0)
            dy = max(a.y0 - origin[1], origin[1] - a.y1, 0)
            if np.hypot(dx, dy) < radius + 4 * pp:
                clear = False
        if clear:
            break
    else:
        raise AssertionError('No free lower-right corner for metric boxes')
    for text, target in zip(metrics, proposed):
        actual = text_box(text, renderer)
        position = ax.transData.transform(text.get_position())
        position += np.array([target.x1 - actual.x1, target.y0 - actual.y0])
        text.set_position(ax.transData.inverted().transform(position))
    labels.extend(metrics)

def verify_layout(fig, ax, legend, labels, renderer):
    lb = legend.get_window_extent(renderer)
    boxes = [(t.get_bbox_patch().get_window_extent(renderer) if t.get_bbox_patch()
              else t.get_window_extent(renderer)) for t in labels]
    names = category_labels(labels)
    right = max(text_box(t, renderer).x1 for t in names)
    assert lb.x1 <= right + 1, 'Legend extends to the right'
    assert all(not lb.overlaps(b) for b in boxes), 'Legend overlaps a label or UB annotation'
    for i, a in enumerate(boxes):
        assert all(not a.overlaps(b) for b in boxes[i+1:]), 'Outer text boxes overlap'
    for b in boxes + [lb] + [t.get_window_extent(renderer) for t in ax.texts]:
        assert b.x0 > 0 and b.y0 > 0 and b.x1 < fig.bbox.width and b.y1 < fig.bbox.height, 'Clipped text'
    # The legend must sit above the circle / data, in the vacant top-right area.
    circle_top = ax.transData.transform((0, 1.055))[1]
    assert lb.y0 > circle_top, 'Legend overlaps the plot'
    metric_texts = [t for t in ax.texts if t.get_text() in METRIC_LABELS]
    assert len(metric_texts) == 2
    assert not any(t.get_text() in ['DSC (%)  ↑', 'HD95 (mm)  ↓'] for t in ax.texts), 'Old heading remains'
    for text in metric_texts:
        b = text_box(text, renderer)
        assert b.x1 <= right + 1
        assert text.get_fontsize() == 14
    errors = []
    for i, text in enumerate(labels):
        if not text.get_text().startswith('UB '):
            continue
        b = text_box(labels[i - 1], renderer)
        ub = text.get_window_extent(renderer)
        error = abs((b.x0 + b.x1 - ub.x0 - ub.x1) / 2)
        errors.append(error)
        assert error <= 1, ('UB center offset', text.get_text(), error)
        upper = labels[i - 1].get_position()[1] > 0
        gap = (ub.y0 - b.y1) if upper else (b.y0 - ub.y1)
        assert abs(gap - 4 * fig.dpi / 72) < 1, ('UB gap', text.get_text(), gap)
    assert len(errors) in (0, 18)
    assert all(abs(np.linalg.norm(t.get_position()) - 1.105) < 1e-10 and t.get_fontsize() == 14 for t in names)
    print('LAYOUT PASS:', 'UBs=', len(errors), 'max_center_error_px=', max(errors, default=0),
          'metric_boxes=2; category anchors and sizes unchanged', flush=True)

def apply_depth_fills(ax, methods):
    """Rank distinct polygon areas; change only fill alpha, never artist order."""
    filled = [m for m in methods if m != 'Joint-Baseline']
    patches = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(patches) == len(filled)
    before = [p.get_xy().copy() for p in patches]
    areas = [float(abs(np.dot(v[:,0], np.roll(v[:,1],-1)) -
                       np.dot(v[:,1], np.roll(v[:,0],-1))) / 2) for v in before]
    distinct = sorted(set(areas))
    alpha = {area: float(value) for area, value in zip(distinct, np.linspace(.16,.07,len(distinct)))}
    for method, patch, area in zip(filled, patches, areas):
        patch.set_alpha(alpha[area])
        print(f'  FILL {method}: area={area:.8f}, alpha={alpha[area]:.3f}', flush=True)
    assert all(np.array_equal(p.get_xy(), v) for p,v in zip(patches,before))
    assert all(alpha[a] > alpha[b] for a,b in zip(distinct, distinct[1:]))
    assert not distinct or np.isclose(alpha[distinct[0]], .16)
    assert len(distinct) < 2 or np.isclose(alpha[distinct[-1]], .07)
    assert [p for p in ax.patches if isinstance(p, Polygon)] == patches


def plot(family, dataset=None, expand=False, depth=False, return_figure=False):
    lesions = dataset is None
    items = [(ds,4,ds) for ds in DATASETS] if lesions else [(dataset,k,label) for k,label in STRUCTURES]
    count = len(items)
    # Clockwise traversal: top left -> top right -> bottom right -> bottom left.
    # A category and its two metrics are mirror paired across the horizontal diameter.
    top_angles = np.pi - (np.arange(count)+.5)*np.pi/count
    axes = [(item,'DSC',angle) for item,angle in zip(items,top_angles)]
    axes += [(item,'HD95',-angle) for item,angle in zip(items[::-1],top_angles[::-1])]
    assert all(abs(math.cos(top_angles[i])-math.cos(-top_angles[i])) < 1e-10 for i in range(count))
    methods = [m for m in METHODS[family] if all(sourcekey(family,item,m,metric) in DATA for item,metric,_ in axes)]
    partial = [m for m in METHODS[family] if m not in methods and any(sourcekey(family,item,m,metric) in DATA for item,metric,_ in axes)]
    assert not partial or (lesions and family=='main' and partial==['jSTABL'])
    scales = build_scales(family, dataset, axes, methods, partial, expand)
    target_count = sum(s.factor == 2 for s in scales)
    STATS['expanded_axes'] += sum(s.factor>1 for s in scales)
    STATS['target_axes'] += target_count
    fig = plt.figure(figsize=(8.6,7.8), dpi=240)
    ax = fig.add_axes([.04,.04,.92,.92])
    ax.set(xlim=(-1.34,1.34),ylim=(-1.34,1.34),aspect='equal')
    ax.axis('off'); ax.set_xticks([]); ax.set_yticks([])
    for r in [.25,.5,.75,1]:
        ax.add_patch(Circle((0,0),r,fill=False,edgecolor='#D8DAD8',linewidth=.6,zorder=0))
    for _,_,theta in axes:
        ax.plot([0,1.06*math.cos(theta)],[0,1.06*math.sin(theta)],color='#D2D5D2',lw=.55,zorder=0)
    # A faint diameter separates metric headings without cutting any data polygon.
    ax.plot([-1.015,1.015],[0,0],color='#C6CAC6',lw=.55,zorder=0)

    vertices=[]
    for method in methods:
        radii = [s.radius(DATA[sourcekey(family,item,method,metric)][0]) for s,(item,metric,_) in zip(scales,axes)]
        assert min(radii) >= 0 and max(radii) <= 1.055
        pts=np.array([[r*math.cos(theta),r*math.sin(theta)] for r,(_,_,theta) in zip(radii,axes)])
        pts=np.vstack([pts,pts[0]])
        assert np.array_equal(pts[0],pts[-1])
        ours=method.endswith('Ours'); color=COLORS[method]
        if method != 'Joint-Baseline':
            ax.add_patch(Polygon(pts[:-1],closed=True,facecolor=color,edgecolor='none',alpha=.095 if ours else .065,zorder=1))
        ax.plot(pts[:,0],pts[:,1],color=color,linewidth=1.4 if ours else 1.0,alpha=1,zorder=3 if ours else 2)
        vertices.extend(pts[:-1])
        for p,(item,metric,_) in zip(pts,axes):
            key=sourcekey(family,item,method,metric)
            STATS['source_cells'].add(key)
            if is_reference(family,item[0],method,item[1]):
                ax.scatter(*p,s=14,facecolor='white',edgecolor=color,linewidth=.7,zorder=6)
        STATS['closed_curves']+=1; STATS['ordinary_points']+=len(axes)
    for method in partial:
        for s,(item,metric,theta) in zip(scales,axes):
            key=sourcekey(family,item,method,metric)
            if key not in DATA:
                continue
            r=s.radius(DATA[key][0]); assert 0<=r<=1.055
            ax.scatter(r*math.cos(theta),r*math.sin(theta),marker='x',s=18,linewidth=.9,color=COLORS[method],zorder=6)
            vertices.append([r*math.cos(theta),r*math.sin(theta)])
            STATS['partial_points']+=1; STATS['source_cells'].add(key)

    tick_texts=[]; other_texts=[]
    for ai,(s,(item,metric,theta)) in enumerate(zip(scales,axes)):
        unit=np.array([math.cos(theta),math.sin(theta)])
        tangent=np.array([-unit[1],unit[0]])
        for ri,r in enumerate([.25,.5,.75,1]):
            value=s.value(r)
            assert abs(s.radius(value)-r)<1e-8
            # Offset from the spoke; horizontal type, no opaque white rectangles.
            p=r*unit
            text=ax.text(*p,s.label(value),fontsize=8.0,ha='center',va='center',color='#202020',zorder=8)
            tick_texts.append((text,p,unit,tangent))
            STATS['decimal_ticks' if s.decimals else 'integer_ticks']+=1
        ds,label,name=item
        face,edge=METRIC_COLORS[metric]
        anchor=unit*1.105
        # Side labels grow away from the circle. Top/bottom labels remain centered.
        ha='left' if unit[0]>.65 else 'right' if unit[0]<-.65 else 'center'
        labeltext=ax.text(*anchor,name,fontsize=14,ha=ha,va='center',color='#161616',zorder=9,
                         bbox=dict(boxstyle='round,pad=.24,rounding_size=.22',fc=face,ec=edge,lw=.8))
        other_texts.append(labeltext)
        if lesions and family=='main':
            key=sourcekey(family,item,'Joint Upper Bound',metric)
            assert key in DATA
            v=DATA[key][0]
            ub='UB '+(f'{v:.1f}' if metric=='DSC' else f'{v:.2f}')
            # Place above the upper labels / below the lower labels, away from data ticks.
            txt=ax.annotate(ub,xy=anchor,xytext=(0,22 if metric=='DSC' else -22),textcoords='offset points',fontsize=10,
                            ha=ha,va='center',color='#646464',zorder=10)
            other_texts.append(txt)
            STATS['upper_labels']+=1; STATS['source_cells'].add(key)

    title=FAMILY_TITLE[family]+'  |  '+('Lesions' if lesions else dataset+' healthy tissues')
    handles=[Line2D([],[],color=COLORS[m],linewidth=1.0,label=m) if m=='Joint-Baseline'
             else Patch(facecolor=COLORS[m]+'45',edgecolor=COLORS[m],linewidth=.8,label=m) for m in methods]
    for m in partial:
        handles.append(Line2D([],[],color=COLORS[m],marker='x',linestyle='none',markersize=5.5,label=m+' (available only)'))
    leg=fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.985,.975),ncol=2,
                   frameon=False,fontsize=11,handlelength=1.5,handletextpad=.4,columnspacing=.5,labelspacing=.35,borderaxespad=0)

    # Full spokes stay continuous. Increase drawing space if a dense ring needs it.
    pixels_per_pt=fig.dpi/72
    gap=1.2*pixels_per_pt
    normals=np.array([[-math.sin(theta),math.cos(theta)] for _,_,theta in axes]+[[0,1]])
    base_offset=gap+.3*pixels_per_pt
    layout_ok=False
    for enlargement in [1.]:
        set_fixed_geometry(fig, ax, (family, dataset), BASELINE_GEOMETRY)
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        position_legend(fig, ax, leg, other_texts, renderer)
        fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        origin_px=ax.transData.transform([0,0])
        vertex_px=ax.transData.transform(vertices)
        other_boxes=[(t.get_bbox_patch().get_window_extent(renderer) if t.get_bbox_patch() else t.get_window_extent(renderer)).padded(gap)
                     for t in other_texts]
        other_boxes.append(leg.get_window_extent(renderer).padded(gap))
        placed=[]; chosen=[]
        if expand:
            fixed_positions = FROZEN_TICKS[(family,dataset)]
            for i, (text, _, _, _) in enumerate(tick_texts):
                text.set_position(fixed_positions[i])
            fig.canvas.draw(); renderer=fig.canvas.get_renderer()
            placed = [text.get_window_extent(renderer) for i,(text,_,_,_) in enumerate(tick_texts)
                      if scales[i//4].factor == 1]
        for start in range(0,len(tick_texts),4):
            if expand and scales[start//4].factor == 1:
                for text,_,_,_ in tick_texts[start:start+4]:
                    chosen.append((text,text.get_position(),text.get_window_extent(renderer),0))
                continue
            if expand:
                # Keep each updated tick near its original position and side.
                # Search the smallest displacement first; never move an
                # untouched axis just to make room for an updated label.
                offsets = sorted(
                    [(a,b) for a in np.arange(-6,6.5,.5)
                     for b in np.arange(-8,8.5,.5)],
                    key=lambda pair: (pair[0]**2+pair[1]**2,abs(pair[0]),pair[0],pair[1]))
                for index in range(start,start+4):
                    text,p,unit,tangent = tick_texts[index]
                    oldpos = ax.transData.transform(fixed_positions[index])
                    spoke = ax.transData.transform(p)
                    oldside = np.sign(np.dot(oldpos-spoke,tangent))
                    for radial_shift,tangent_shift in offsets:
                        pos = oldpos + (unit*radial_shift+tangent*tangent_shift)*pixels_per_pt
                        if np.sign(np.dot(pos-spoke,tangent)) != oldside:
                            continue
                        text.set_position(ax.transData.inverted().transform(pos))
                        box = text.get_window_extent(renderer)
                        corners = np.array([[box.x0,box.y0],[box.x0,box.y1],
                                            [box.x1,box.y0],[box.x1,box.y1]])
                        distances = (corners-origin_px) @ normals.T
                        spokes_clear = np.all((distances.min(axis=0)>=gap+.275*pixels_per_pt)|
                                              (distances.max(axis=0)<=-gap-.275*pixels_per_pt))
                        dx = np.maximum(np.maximum(box.x0-vertex_px[:,0],vertex_px[:,0]-box.x1),0)
                        dy = np.maximum(np.maximum(box.y0-vertex_px[:,1],vertex_px[:,1]-box.y1),0)
                        vertices_clear = np.all(np.hypot(dx,dy)>=gap+.7*pixels_per_pt)
                        if (spokes_clear and vertices_clear
                                and all(not box.padded(.5*pixels_per_pt).overlaps(prev) for prev in placed)
                                and all(not box.overlaps(prev) for prev in other_boxes)):
                            placed.append(box)
                            chosen.append((text,text.get_position(),box,
                                           math.hypot(radial_shift,tangent_shift)))
                            break
                    else:
                        raise AssertionError((title,'No nearby tick position',index,text.get_text()))
                continue
            options=[]
            for side in [1,-1]:
                trial=[]
                for text,p,unit,tangent in tick_texts[start:start+4]:
                    direction=tangent*side
                    b=text.get_window_extent(renderer)
                    halfwidth=(abs(direction[0])*b.width+abs(direction[1])*b.height)/2
                    for radial_shift, extra in [(shift, ext) for shift in [0,2,-2,4,-4,6,-6] for ext in np.arange(0,22.5,.5)]:
                        pos=ax.transData.transform(p)+direction*(halfwidth+base_offset+extra*pixels_per_pt)+unit*radial_shift*pixels_per_pt
                        text.set_position(ax.transData.inverted().transform(pos))
                        box=text.get_window_extent(renderer)
                        corners=np.array([[box.x0,box.y0],[box.x0,box.y1],[box.x1,box.y0],[box.x1,box.y1]])
                        distances=(corners-origin_px) @ normals.T
                        spokes_clear=np.all((distances.min(axis=0)>=gap+.275*pixels_per_pt)|
                                            (distances.max(axis=0)<=-gap-.275*pixels_per_pt))
                        dx=np.maximum(np.maximum(box.x0-vertex_px[:,0],vertex_px[:,0]-box.x1),0)
                        dy=np.maximum(np.maximum(box.y0-vertex_px[:,1],vertex_px[:,1]-box.y1),0)
                        vertices_clear=np.all(np.hypot(dx,dy)>=gap+.7*pixels_per_pt)
                        if (spokes_clear and vertices_clear
                                and all(not box.padded(.5*pixels_per_pt).overlaps(prev) for prev in placed+[x[2] for x in trial])
                                and all(not box.overlaps(prev) for prev in other_boxes)):
                            trial.append((text,text.get_position(),box,extra)); break
                    else:
                        break
                if len(trial)==4:
                    options.append((max(x[3] for x in trial)*10+sum(x[3] for x in trial)+(side<0)*.01,trial))
            if not options:
                break
            _,trial=min(options,key=lambda item:item[0])
            for text,pos,box,extra in trial:
                text.set_position(pos)
                placed.append(box); chosen.append((text,pos,box,extra))
        else:
            layout_ok=True; break
    assert layout_ok, ('No continuous-spoke tick layout',title,start)
    moved=sum(x[3]>0 for x in chosen)
    max_extra=max(x[3] for x in chosen)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for text in [x[0] for x in tick_texts]+other_texts:
        b=text.get_window_extent(renderer)
        assert b.x0>0 and b.y0>0 and b.x1<fig.bbox.width and b.y1<fig.bbox.height,(title,'text outside canvas',text.get_text())
    texts=[t for t in ax.texts if t.get_visible()]
    collisions=[]
    for i,a in enumerate(texts):
        ab=a.get_window_extent(renderer)
        for b in texts[i+1:]:
            if ab.overlaps(b.get_window_extent(renderer)):
                collisions.append((a.get_text(),b.get_text()))
    assert not collisions,(title,'overlapping text',collisions)
    # Metric/category pairs share x coordinates and exactly match manuscript lookup keys.
    for i in range(count):
        a,b=axes[i],axes[-1-i]
        assert a[0]==b[0] and a[1]=='DSC' and b[1]=='HD95'
    name=FAMILY_FILE[family]+('_lesion' if lesions else '_healthy_tissue_'+dataset)
    assert not any(leg.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in texts)
    canvas=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB')
    content=ImageChops.difference(canvas,Image.new('RGB',canvas.size,'white')).getbbox()
    assert content and content[0]>0 and content[1]>0 and content[2]<canvas.width and content[3]<canvas.height
    cropped=ImageOps.expand(canvas.crop(content),border=5,fill='white')
    verify_layout(fig, ax, leg, other_texts, renderer)
    original = cropped
    if depth:
        apply_depth_fills(ax, methods)
        fig.canvas.draw()
        changed_canvas = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB')
        assert ImageChops.difference(changed_canvas, Image.new('RGB', changed_canvas.size, 'white')).getbbox() == content
        cropped = ImageOps.expand(changed_canvas.crop(content), border=5, fill='white')
    if expand:
        fixed_positions = FROZEN_TICKS[(family,dataset)]
        for i,(text,_,_,_) in enumerate(tick_texts):
            if scales[i//4].factor == 1:
                assert np.array_equal(text.get_position(),fixed_positions[i]), (name,'unchanged tick moved',i)
    else:
        FROZEN_TICKS[(family,dataset)] = [tuple(text.get_position()) for text,_,_,_ in tick_texts]
    STATS['figures']+=1
    print(name, 'PASS',f'axes={len(axes)}, expanded={sum(s.factor>1 for s in scales)}, target_axes={target_count}, canvas_scale={enlargement}, png={cropped.size}, closed={len(methods)}, partial={len(partial)}, tick_gap={base_offset/pixels_per_pt:.2f}pt, extra_shifts={moved}, max_extra={max_extra:.1f}pt',flush=True)
    if return_figure:
        return fig, ax, leg, content
    plt.close(fig)
    return (original, cropped) if depth else cropped


LAYOUTS = {('01', 'lesion'): {'columns': [(1343, 1643, 4), (1662, 2109, 4)],
                    'y': 5,
                    'rows': 4,
                    'top_boundary': 197,
                    'right_boundary': 2127},
 ('01', 'tissue'): {'columns': [(851, 1152, 3), (1170, 1522, 3)],
                    'y': 5,
                    'rows': 3,
                    'top_boundary': 161,
                    'right_boundary': 1540},
 ('02', 'lesion'): {'columns': [(1589, 1857, 3), (1874, 2100, 2)],
                    'y': 5,
                    'rows': 3,
                    'top_boundary': 86,
                    'right_boundary': 2119},
 ('02', 'tissue'): {'columns': [(1010, 1279, 3), (1296, 1522, 2)],
                    'y': 5,
                    'rows': 3,
                    'top_boundary': 161,
                    'right_boundary': 1540},
 ('03', 'lesion'): {'columns': [(1516, 1829, 1), (1848, 2100, 1)],
                    'y': 18,
                    'rows': 1,
                    'top_boundary': 5,
                    'right_boundary': 2119},
 ('03', 'tissue'): {'columns': [(937, 1250, 1), (1270, 1522, 1)],
                    'y': 5,
                    'rows': 1,
                    'top_boundary': 67,
                    'right_boundary': 1540}}

def ink_box(im):
    return ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()

def arrange_legends(original, name):
    path = Path(name)
    key = path.name[:2], 'tissue' if 'healthy_tissue' in path.name else 'lesion'
    cfg = LAYOUTS[key]
    original = original.convert('RGB')
    cols = cfg['columns']
    old_box = (cols[0][0], cfg['y'], cols[-1][1], cfg['y']+(cfg['rows']-1)*47+34)
    entries = []
    for x0,x1,count in cols:
        for row in range(count):
            cell = original.crop((x0,cfg['y']+row*47,x1,cfg['y']+row*47+34))
            b = ink_box(cell)
            if b:
                entries.append(cell.crop(b))
    old_count = np.any(np.asarray(original.crop(old_box))<255,axis=2).sum()
    assert sum(np.any(np.asarray(e)<255,axis=2).sum() for e in entries)==old_count
    if key == ('01','lesion'):
        old_entry = entries[-1]
        marker = old_entry.crop((0,4,21,25))
        label = old_entry.crop((51,0,old_entry.width,old_entry.height))
        aligned = Image.new('RGB',(72+label.width,old_entry.height),'white')
        aligned.paste(marker,(19,4))
        aligned.paste(label,(72,0))
        assert np.any(np.asarray(aligned)<255,axis=2).sum()==np.any(np.asarray(old_entry)<255,axis=2).sum()
        entries[-1] = aligned
    scale = 1.
    placements = []
    if key == ('01','tissue'):
        groups = ([0,1,4],[2,3,5]) if len(entries)==6 else ([0,1,3],[2,4])
        scale = .91
        for col,group in enumerate(groups):
            for row,index in enumerate(group):
                placements.append((index, 1032+330*col, 165+35*row))
    elif key == ('02','tissue'):
        for col,group in enumerate(([0,1,2],[3,4])):
            for row,index in enumerate(group):
                placements.append((index, 1029+281*col, 165+41*row))
    elif key == ('01','lesion'):
        placements = [(i,1653,209+47*i) for i in range(len(entries))]
    elif key == ('02','lesion'):
        for col,group in enumerate(([0,1,2],[3,4])):
            for row,index in enumerate(group):
                placements.append((index,1573+285*col,98+47*row))
    elif key == ('03','lesion'):
        placements = [(0,1500,75),(1,1832,75)]
    else:
        placements = [(0,1192,79),(1,1192,126)]
    assert sorted(i for i,_,_ in placements)==list(range(len(entries)))
    background = original.copy()
    background.paste('white', old_box)
    result = background.copy()
    new_boxes = []
    for i,x,y in placements:
        entry = entries[i]
        if scale != 1:
            entry = entry.resize((round(entry.width*scale),round(entry.height*scale)),Image.Resampling.LANCZOS)
        box = (x,y,x+entry.width,y+entry.height)
        assert y>=cfg['top_boundary'] and box[2]<=cfg['right_boundary'], (path.name,'legend out of label bounds',box)
        assert box[3]<=original.height
        assert ink_box(background.crop((x-8,y-8,box[2]+8,box[3]+8))) is None, (path.name,'background collision',box)
        for a,b,c,d in new_boxes:
            assert box[2]<=a or x>=c or box[3]<=b or y>=d, (path.name,'legend entry overlap')
        result.paste(entry,(x,y))
        new_boxes.append(box)

    changed = np.any(np.asarray(result)!=np.asarray(original),axis=2)
    allowed = np.zeros(changed.shape,dtype=bool)
    for x0,y0,x1,y1 in [old_box]+new_boxes:
        allowed[y0:y1,x0:x1] = True
    assert not np.any(changed & ~allowed), (path.name,'non-legend content changed')
    if key==('01','lesion'):
        region=(917,154,1030,178)
        assert ImageChops.difference(original.crop(region),result.crop(region)).getbbox() is None
    crop = ink_box(result)
    final = result.crop(crop)
    assert ink_box(final)==(0,0,*final.size)
    print(f'{path.name}: LEGEND PASS | top={min(b[1] for b in new_boxes)}, '
          f'right={max(b[2] for b in new_boxes)} | png={final.size}',flush=True)
    return final



def filename(family, dataset):
    return FAMILY_FILE[family] + ('_lesion' if dataset is None else '_healthy_tissue_' + dataset) + '.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    assert source != output, 'Output directory must differ from source PNG directory'
    assert output == OUT or not output.exists(), 'Use the script directory or a new output folder'
    ast.parse(Path(__file__).read_text(encoding='utf-8'))
    jobs = [(family, ds) for family in METHODS for ds in [None] + ANATOMY_DATASETS]
    expected = {filename(*job) for job in jobs}
    assert {p.name for p in source.glob('*.png')} == expected
    assert sum(map(len, ADJUST_AXES.values())) == 29
    output.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        name = filename(*job)
        expand = job in JOBS
        if expand:
            plot(*job, expand=False)
        baseline, adjusted = plot(*job, expand=expand, depth=True)
        baseline = arrange_legends(baseline, name)
        adjusted = arrange_legends(adjusted, name)
        with Image.open(source / name) as opened:
            previous = opened.convert('RGB')
        assert baseline.size == previous.size, (name, 'source dimensions changed', baseline.size, previous.size)
        assert ImageChops.difference(baseline, previous).getbbox() is None, (name, 'source reconstruction mismatch')
        assert adjusted.size == previous.size
        assert ImageChops.difference(adjusted, baseline).getbbox() is not None
        adjusted.save(output / name, dpi=(240,240))
        with Image.open(output / name) as saved:
            assert saved.format == 'PNG' and saved.size == previous.size
        print('SAVED', name, flush=True)
    assert {p.name for p in output.glob('*.png')} == expected
    assert STATS['target_axes'] == 29
    assert STATS['source_cells'] == set(DATA)
    print('PASS: 15 PNGs regenerated; area-ranked alpha 0.16 -> 0.07; 29 doubled axes preserved; Joint-Baseline unfilled.',flush=True)



# ===== Minimal composition layer; the original plotting functions above are retained. =====
def export_original_panel(family, dataset):
    import io
    from matplotlib.transforms import Bbox
    expand = (family, dataset) in JOBS
    if expand:
        plot(family, dataset, expand=False)
    fig, ax, legend, content = plot(family, dataset, expand=expand, depth=True, return_figure=True)
    fig.canvas.draw()
    before = np.asarray(fig.canvas.buffer_rgba()).copy()
    reference = OUT.parent / 'redesigned_legend_only_labels_inside_local_axes_2x_dark_to_light' / filename(family, dataset)
    if reference.exists():
        reconstructed = ImageOps.expand(Image.fromarray(before).convert('RGB').crop(content), border=5, fill='white')
        reconstructed = arrange_legends(reconstructed, reference.name)
        with Image.open(reference) as original_image:
            assert reconstructed.size == original_image.size
            assert ImageChops.difference(reconstructed, original_image.convert('RGB')).getbbox() is None, 'Original reconstruction differs'
    renderer = fig.canvas.get_renderer()
    decoration_boxes = [legend.get_window_extent(renderer)]
    for t in ax.texts:
        if t.get_bbox_patch() is not None:
            decoration_boxes.append(t.get_bbox_patch().get_window_extent(renderer))
    retained = [t for t in ax.texts if t.get_text() not in METRIC_LABELS]
    text_state = [(t, t.get_text(), t.get_position(), t.get_fontsize(), t.get_color()) for t in retained]
    line_state = [(line, line.get_xydata().copy(), line.get_linewidth(), line.get_color()) for line in ax.lines]
    # Only the requested decorations are changed, after the original layout is complete.
    legend.remove()
    removed_boxes = 0
    for t in list(ax.texts):
        if t.get_text() in METRIC_LABELS:
            t.remove()
        elif t.get_text() in {name for _, name in STRUCTURES}:
            t.set_bbox(None)
            metric = 'DSC' if t.get_position()[1] > 0 else 'HD95'
            t.set_color(METRIC_COLORS[metric][1])
            removed_boxes += 1
    assert removed_boxes == 12
    fig.canvas.draw()
    after = np.asarray(fig.canvas.buffer_rgba()).copy()
    allowed = np.zeros(before.shape[:2], dtype=bool)
    for box in decoration_boxes:
        x0, x1 = max(0, math.floor(box.x0)-3), min(before.shape[1], math.ceil(box.x1)+3)
        y0, y1 = max(0, before.shape[0]-math.ceil(box.y1)-3), min(before.shape[0], before.shape[0]-math.floor(box.y0)+3)
        allowed[y0:y1, x0:x1] = True
    assert np.array_equal(before[~allowed], after[~allowed]), 'Content outside removed decorations changed'
    for t, text, pos, size, color in text_state:
        if text in {name for _, name in STRUCTURES}:
            color = METRIC_COLORS['DSC' if pos[1] > 0 else 'HD95'][1]
        assert (t.get_text(), t.get_position(), t.get_fontsize(), t.get_color()) == (text, pos, size, color)
    for line, xy, lw, color in line_state:
        assert np.array_equal(line.get_xydata(), xy) and line.get_linewidth() == lw and line.get_color() == color
    if family == 'ablation':
        # Remove only Baseline's fill; leave all other methods' opacity unchanged.
        fills = [p for p in ax.patches if isinstance(p, Polygon)]
        assert len(fills) == len(METHODS[family])
        baseline_fill = fills[METHODS[family].index('Baseline')]
        baseline_fill.remove()
        assert len([p for p in ax.patches if isinstance(p, Polygon)]) == len(fills)-1
    # Remove the now-empty legend margin without changing the radar geometry.
    x0, y0, x1, y1 = ImageChops.difference(Image.fromarray(after).convert('RGB'),
                                         Image.new('RGB', (after.shape[1], after.shape[0]), 'white')).getbbox()
    x0, y0, x1, y1 = x0-28, y0-28, x1+28, y1+28
    bbox = Bbox.from_extents(x0/fig.dpi, (before.shape[0]-y1)/fig.dpi,
                            x1/fig.dpi, (before.shape[0]-y0)/fig.dpi)
    png, pdf = io.BytesIO(), io.BytesIO()
    fig.savefig(png, format='png', dpi=600, bbox_inches=bbox, pad_inches=0)
    fig.savefig(pdf, format='pdf', bbox_inches=bbox, pad_inches=0)
    png.seek(0)
    im = Image.open(png).convert('RGB')
    plt.close(fig)
    print('MINIMAL EDIT PASS:', family, dataset, 'metric label colors and original text/curve properties verified; Baseline fill removed for ablation only', flush=True)
    return im, pdf.getvalue()


def make_shared_footer(family):
    import io
    fig = plt.figure(figsize=(7.2, .36), dpi=600)
    for i, ds in enumerate(ANATOMY_DATASETS):
        fig.text((i+.5)/4, .73, f'({chr(97+i)}) {ds}', ha='center', va='center',
                 fontsize=8, fontfamily='Times New Roman', color='black')
    # Both metric explanations and the method legend share one centered row.
    row_y = .25
    metric_texts = [fig.text(0, row_y, label, ha='left', va='center', fontsize=7,
                             color=METRIC_COLORS[metric][1])
                    for label, metric in zip(METRIC_LABELS, ['DSC', 'HD95'])]
    methods = [m for m in METHODS[family]
               if any(all((family, ds, m, label, metric) in DATA
                          for label, _ in STRUCTURES for metric in ['DSC', 'HD95'])
                      for ds in ANATOMY_DATASETS)]
    handles = [Line2D([], [], color=COLORS[m], linewidth=1.0, label=m)
               if m == 'Joint-Baseline' or (family == 'ablation' and m == 'Baseline')
               else Patch(facecolor=COLORS[m]+'45', edgecolor=COLORS[m], linewidth=.8, label=m)
               for m in methods]
    legend = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0, row_y),
                        ncol=len(methods), frameon=False, fontsize=8, handlelength=1.5,
                        handletextpad=.4, columnspacing=.8, borderaxespad=0, borderpad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer).width for t in metric_texts]
    metric_gap, legend_gap = 8*fig.dpi/72, 12*fig.dpi/72
    total_width = sum(widths) + metric_gap + legend_gap + legend.get_window_extent(renderer).width
    assert total_width < fig.bbox.width, 'Shared row exceeds figure width'
    left = (fig.bbox.width-total_width)/2
    metric_texts[0].set_x(left/fig.bbox.width)
    metric_texts[1].set_x((left+widths[0]+metric_gap)/fig.bbox.width)
    legend.set_bbox_to_anchor(((left+sum(widths)+metric_gap+legend_gap)/fig.bbox.width, row_y))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Matplotlib places legend handles above the text-box center by default.
    # Align the actual line/patch centers, rather than only the legend container.
    target_y = metric_texts[0].get_window_extent(renderer).y0 + metric_texts[0].get_window_extent(renderer).height/2
    for handle in legend.legend_handles:
        box = handle.get_window_extent(renderer)
        delta_px = target_y - (box.y0+box.y1)/2
        transform = handle.get_transform() if isinstance(handle, Line2D) else handle.get_data_transform()
        inverse = transform.inverted()
        delta_y = inverse.transform((0, delta_px))[1] - inverse.transform((0, 0))[1]
        if isinstance(handle, Line2D):
            handle.set_ydata(np.asarray(handle.get_ydata()) + delta_y)
        else:
            handle.set_y(handle.get_y() + delta_y)
    for text in legend.get_texts():
        box = text.get_window_extent(renderer)
        delta_px = target_y - (box.y0+box.y1)/2
        inverse = text.get_transform().inverted()
        delta_y = inverse.transform((0, delta_px))[1] - inverse.transform((0, 0))[1]
        text.set_y(text.get_position()[1] + delta_y)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_artists = metric_texts + list(legend.get_texts()) + list(legend.legend_handles)
    row_centers = [(a.get_window_extent(renderer).y0+a.get_window_extent(renderer).y1)/2 for a in row_artists]
    assert max(row_centers)-min(row_centers) < .01, 'Shared row centers differ'
    boxes = [t.get_window_extent(renderer) for t in fig.texts] + [legend.get_window_extent(renderer)]
    assert all(b.x0>=0 and b.x1<=fig.bbox.width and b.y0>=0 and b.y1<=fig.bbox.height for b in boxes)
    assert all(not a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
    png, pdf = io.BytesIO(), io.BytesIO()
    fig.savefig(png, format='png', dpi=600)
    fig.savefig(pdf, format='pdf')
    png.seek(0)
    im = Image.open(png).convert('RGB')
    plt.close(fig)
    return im, pdf.getvalue()


def combined_main():
    import fitz
    for family in METHODS:
        panels = [export_original_panel(family, ds) for ds in ANATOMY_DATASETS]
        assert len({im.size for im, _ in panels}) == 1, 'Original panel canvases differ'
        panel_width = 1080
        im0 = panels[0][0]
        panel_height = round(im0.height * panel_width / im0.width)
        footer, footer_pdf = make_shared_footer(family)
        width, height = panel_width*4, panel_height+footer.height
        result = Image.new('RGB', (width, height), 'white')
        pdf = fitz.open()
        unit = 72/600
        page = pdf.new_page(width=width*unit, height=height*unit)
        for i, (im, pdf_bytes) in enumerate(panels):
            result.paste(im.resize((panel_width, panel_height), Image.Resampling.LANCZOS), (i*panel_width, 0))
            source = fitz.open(stream=pdf_bytes, filetype='pdf')
            page.show_pdf_page(fitz.Rect(i*panel_width*unit, 0, (i+1)*panel_width*unit, panel_height*unit), source, 0)
            source.close()
        result.paste(footer, (0, panel_height))
        source = fitz.open(stream=footer_pdf, filetype='pdf')
        page.show_pdf_page(fitz.Rect(0, panel_height*unit, width*unit, height*unit), source, 0)
        source.close()
        stem = FAMILY_FILE[family] + '_healthy_tissue_four_panel'
        result.save(OUT/(stem+'.png'), dpi=(600,600))
        pdf.save(OUT/(stem+'.pdf'), garbage=4, deflate=True)
        pdf.close()
        print('SAVED:', stem, result.size, flush=True)


if __name__ == '__main__':
    combined_main()
