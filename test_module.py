import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import medical_data_visualizer as mdv


def test_overweight_column_and_normalization():
    assert "overweight" in mdv.df.columns
    assert mdv.df["cholesterol"].isin([0, 1]).all()
    assert mdv.df["gluc"].isin([0, 1]).all()


def test_draw_cat_plot_returns_facets():
    fig = mdv.draw_cat_plot()
    assert fig is not None
    assert hasattr(fig, "axes")
    assert len(fig.axes.flat) == 2


def test_draw_heat_map_returns_figure():
    fig = mdv.draw_heat_map()
    assert fig is not None
    assert hasattr(fig, "axes")
