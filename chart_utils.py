import matplotlib.pyplot as plt
import matplotlib
import os


class ColoredText:
    ok = {'fontsize': 18, 'color': 'green', 'line_interval_pix': 10}
    warning = {'fontsize': 18, 'color': 'orange', 'line_interval_pix': 10}
    critical = {'fontsize': 18, 'color': 'red', 'line_interval_pix': 10}
    title = {'fontsize': 18, 'color': 'black', 'line_interval_pix': 10}

    @classmethod
    def draw(cls, lines, left_margin=0, **kwargs):
        dpi = matplotlib.rcParams["figure.dpi"]
        fig_height_pix = sum([style['fontsize'] + style['line_interval_pix'] for _,style in lines])
        fig_height_pix += lines[0][1]['line_interval_pix'] # additional space for interval before first line
        fig_height_inch = fig_height_pix / dpi
        char_width_pix = 20  # можно посчитать через max( text.get_window_extent().width по всем строкам) учеть шрифт
        fig_width_pix = max([len(string) + 1 for string,_ in lines]) * char_width_pix
        fig_width_inch = fig_width_pix / dpi
        fig = plt.figure(frameon=False, figsize=(fig_width_inch, fig_height_inch))
        ax = fig.add_axes([0, 0, 1, 1])  # fig_width_pix, fig_height_pix])
        canvas = ax.figure.canvas
        x_pix = left_margin
        y_pix = fig_height_pix
        max_text_width = -1

        for string, style in lines:
            x, y = x_pix / fig_width_pix, (y_pix - style['fontsize'] - style['line_interval_pix']) / fig_height_pix
            text = ax.text(x, y, string, color=style['color'], fontsize=style['fontsize'], **kwargs)
            # print(f'xy_pix=({x_pix,y_pix}), xy=({x},{y})', f'fontsize={line[1]})')
            text.draw(canvas.get_renderer())  # we don't use it. It's need for get_window_extent only.
            text_width = left_margin + text.get_window_extent().width
            max_text_width = max(max_text_width, text_width)
            y_pix -= style['fontsize'] + style['line_interval_pix']
        fig.set_size_inches((max_text_width + 20) / dpi, fig_height_inch, forward=True)
        ax.set_axis_off()
        return fig


def main():
    ct=ColoredText
    lines = [("Поточний стан обладнання:", ct.title),
             ("Поверх 1: Насос 12.11 - ok", ct.ok),
             ("Поверх 1: Насос 12.17 - ok", ct.ok),
             ("Поверх 1: Насос 15.19 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.08 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.09 - ok", ct.ok),
             ("Поверх 2: Насос 18.18 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - перегрів!!!", ct.critical),
             ("Поверх 2: Насос 17.10 - ok", ct.ok),
             ("Поверх 1: Насос 12.11 - ok", ct.ok),
             ("Поверх 1: Насос 12.17 - ok", ct.ok),
             ("Поверх 1: Насос 15.19 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.08 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.09 - ok", ct.ok),
             ("Поверх 2: Насос 18.18 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - перегрів!!!", ct.critical),
             ("Поверх 2: Насос 17.10 - ok", ct.ok),
             ("Поверх 1: Насос 12.11 - ok", ct.ok),
             ("Поверх 1: Насос 12.17 - ok", ct.ok),
             ("Поверх 1: Насос 15.19 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.08 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.09 - ok", ct.ok),
             ("Поверх 2: Насос 18.18 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - перегрів!!!", ct.critical),
             ("Поверх 2: Насос 17.10 - ok", ct.ok),
             ("Поверх 1: Насос 12.11 - ok", ct.ok),
             ("Поверх 1: Насос 12.17 - ok", ct.ok),
             ("Поверх 1: Насос 15.19 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.08 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.09 - ok", ct.ok),
             ("Поверх 2: Насос 18.18 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - перегрів!!!", ct.critical),
             ("Поверх 2: Насос 17.10 - ok", ct.ok),
             ("Поверх 1: Насос 12.11 - ok", ct.ok),
             ("Поверх 1: Насос 12.17 - ok", ct.ok),
             ("Поверх 1: Насос 15.19 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.08 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.09 - ok", ct.ok),
             ("Поверх 2: Насос 18.18 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - перегрів!!!", ct.critical),
             ("Поверх 2: Насос 17.10 - ok", ct.ok),
             ("Поверх 1: Насос 12.11 - ok", ct.ok),
             ("Поверх 1: Насос 12.17 - ok", ct.ok),
             ("Поверх 1: Насос 15.19 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.08 - ok", ct.ok),
             ("Поверх 2: Насос 17.10 - увага", ct.warning),
             ("Поверх 2: Насос 18.09 - ok", ct.ok),
             ("Поверх 2: Насос 18.18 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - перегрів!!!", ct.critical),
             ("Поверх 2: Насос 17.10 - ok", ct.ok),
             ("Поверх 2: Насос 18.08 - ok", ct.ok)]
    fig = ColoredText.draw(lines, 20)
    fig.savefig('../tmp/colored_text.png')
    os.system('telegram-send -i "../tmp/colored_text.png"')
    plt.show()


if __name__ == "__main__":
    main()
