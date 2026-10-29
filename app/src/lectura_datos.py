
import pandas as pd
import os
import sys
import logging
import numpy as np
import arviz as az
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.notebook import tqdm
import cmdstanpy
from cmdstanpy import CmdStanModel
sys.path.append( r'src' )
from utils import update_namespace, read_in3, getOrbit, getRV,getRVSingle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.platypus import Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.platypus import Table, TableStyle
import streamlit as st
from PIL import Image as PILImage

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, Spacer

def lec_type_examples(system):
    df_as, df_v1, df_v2, df_priors = read_in3('data/{}.in3'.format(system))
    print('V1 observations:')
    return df_v1, df_v2



def lec_type_star02(path):
    df_v1 = pd.read_csv(path, delimiter='\t')
    df_v1 = df_v1.rename(columns={'#BJD (d)': 'epoch', 'RV (km/s)': 'RV', 'e_RV (km/s)': 'err'})
    print(df_v1.columns)
    df_v1.drop('Spectrograph', axis=1, inplace=True)
    df_v1
    return df_v1
def lec_type_star01(path):
    df_v1 = pd.read_csv(path, delimiter=' ')
    df_v1 = df_v1.rename(columns={'#HJD': 'epoch', 'RV': 'RV', 'err': 'err'})
    return df_v1

