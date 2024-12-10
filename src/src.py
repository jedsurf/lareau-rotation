import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def legend_kwargs():
    # Commonly used kwargs for figure legends
    kwargs = {
        "frameon" : False, 
        "bbox_to_anchor" : (1, 0.5), 
        "loc" : "center left"}
    return kwargs