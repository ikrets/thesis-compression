import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import cv2
import numpy as np


def figure_to_numpy(fig):
    with BytesIO() as fp:
        fig.savefig(fp, format='png')
        fp.seek(0)
        img_arr = np.frombuffer(fp.getvalue(), dtype=np.uint8)
        img_arr = cv2.imdecode(img_arr, 1)
        plt.close(fig)

    return img_arr


def lambda_vs_bitrate_and_psnr(lambdas, bitrates, psnrs, **kwargs):
    fig, axes = plt.subplots(1, 2, **kwargs)

    im = axes[0].hexbin(lambdas, psnrs, cmap='inferno', bins='log')
    cb = plt.colorbar(im, ax=axes[0])
    cb.set_label('Count')
    axes[0].set_xlabel('$\lambda$')
    axes[0].set_ylabel('PSNR')

    im = axes[1].hexbin(lambdas, bitrates, cmap='inferno', bins='log')
    cb = plt.colorbar(im, ax=axes[1])
    cb.set_label('Count')
    axes[1].set_xlabel('$\lambda$')
    axes[1].set_ylabel('bitrate')

    plt.tight_layout()

    return fig


def rate_distortion_curve(compression_eval_df,
                          downstream_metrics,
                          **kwargs):
    fig, axes = plt.subplots(1 + len(downstream_metrics), 3, **kwargs)

    for row, metric in enumerate(['psnr', *downstream_metrics]):
        for col, parameter in enumerate(['bpp', 'alpha', 'lambda']):
            if parameter == 'alpha':
                hue = 'lambda'
            elif parameter == 'lambda':
                hue = 'alpha'
            elif metric == 'psnr':
                hue = 'alpha'
            else:
                hue = 'lambda'

            p = sns.lineplot(x=parameter, y=metric, hue=hue,
                         marker='o', legend='full',
                         data=compression_eval_df,
                         ax=axes[row, col])
            for t in p.legend().texts:
                t.set_text(t.get_text()[:6])

    plt.tight_layout()
    return fig