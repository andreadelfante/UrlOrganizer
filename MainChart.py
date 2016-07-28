# import sys

from models.Chart import Chart
from models.MetricKey import MetricKey
import plotly.tools as tls

tls.set_credentials_file(username='andreadelfante', api_key='1eob0r0ult')


def main(args):
    direct = args[0]
    left_skipgram = args[1]
    normal_skipgram = args[2]

    chart = Chart(path=direct, left_skipgram=left_skipgram, normal_skipgram=normal_skipgram)
    #chart.plot_linechart(Chart.Metrics.v_measure, "500000", "10", MetricKey.Clustering.KMeans)
    chart.plot_heatmap_window_db(Chart.Metrics.v_measure, Chart.Algorithm.normal_skipgram, MetricKey.Clustering.KMeans, "20")


if __name__ == "__main__":
    # args = sys.argv[1:]
    # main(args=args)
    direct = "/Volumes/AdditionalDriveMAC/Google Drive/1) Tesi/Sperimentazioni/"
    left_skipgram = "Risultati left skipgram.csv"
    normal_skipgram = "Risultati normal skipgram.csv"

    main(args=[direct, left_skipgram, normal_skipgram])
