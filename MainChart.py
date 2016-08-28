# import sys

from models.Chart import Chart
from models.MetricKey import MetricKey
import plotly.tools as tls

tls.set_credentials_file(username='andreadelfante', api_key='1eob0r0ult')


def main(args):
    direct = "/Volumes/AdditionalDriveMAC/Google Drive/1) Tesi/Sperimentazioni/" \
             "cs.illinois.edu/" \
             "ListConstraint/" \
             "intersect/"
    left_skipgram_with_b = "Risultati left skipgram with b intersect.csv"
    left_skipgram_no_b = "Risultati left skipgram no b intersect.csv"
    normal_skipgram = "Risultati normal skipgram intersect.csv"

    chart = Chart(path=direct, left_skipgram_with_b=left_skipgram_with_b, left_skipgram_no_b=left_skipgram_no_b,
                  normal_skipgram=normal_skipgram)
    chart.plot_linechart(Chart.Metrics.v_measure, "100000", "10", MetricKey.Clustering.KMeans)
    #chart.plot_heatmap_window_db(Chart.Metrics.v_measure, Chart.Algorithm.normal_skipgram, MetricKey.Clustering.HDBScan, "20")


if __name__ == "__main__":
    main([])
