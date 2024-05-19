import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def draw_plot(data, title):
    plt.figure(figsize=(15,8))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    sns.lineplot(x=range(len(data)), y=data)

    plt.title(title)
    plt.show()
