import pandas as pd
from nicegui import ui
from scipy.stats import stats

from libs.database import QualityCheckDB

# tolerances, TODO read from config
TOLERANCES = {
    "top_weft_edge": (164, 174),
    "right_warp_edge": (238, 242),
    "bottom_weft_edge": (164, 174),
    "left_warp_edge": (238, 242),
    "front_weft_circle": (4.5, 5.5),
    "back_weft_circle": (4.5, 5.5),
    "front_weft_circle_to_warp_cut": (10, 20),
    "back_weft_circle_to_warp_cut": (10, 20),
    "back_weft_circle_to_weft_cut": (48, 52),
    "circle_to_circle": (158, 162)
}

def plot(title: str,
         df: pd.DataFrame,
         columns: list[str],
         *,
         y_axis_label="",
         label_suffix="",
         tolerances: tuple[float, float] = None):
    with ui.matplotlib(figsize=(8, 5), layout='constrained').figure as fig:
        fig.suptitle(title)
        ax = fig.gca()
        ax.set_xlabel("Zeit")
        ax.set_ylabel(y_axis_label)
        if tolerances is not None:
            ax.axhspan(ymin=tolerances[0], ymax=tolerances[1], color="tab:green", alpha=0.1)
        for column in columns:
            ax.plot(df[column], ".", label=column + label_suffix)
        ax.legend()
        ax.grid()



class TrendPlots:
    def __init__(self):
        self.qc_db = QualityCheckDB()
        self.selection = 1
        self.choices = {
            1: {
                "name": "Letzte Stunde",
                "query": "-1 hour",
            },
            2: {
                "name": "Letzter Tag",
                "query": "-1 day",
                "resample": "1min"
            },
            3: {
                "name": "Letzte Woche",
                "query": "-7 days",
                "resample": "1h"
            }
        }

        self.dialog = ui.dialog()

        ui_choices = {}
        for key, value in self.choices.items():
            ui_choices[key] = value["name"]

        self.ui_upper()
        ui.toggle(ui_choices, on_change=self._on_plot_toggle).bind_value(self, 'selection')
        self.ui_lower()

    async def _on_plot_toggle(self):
        self.ui_lower.refresh()

    async def _on_table_row_click(self, event):
        try:
            row_data = event.args[1]
            image = self.qc_db.retrieve_image(row_data["id"])

            self.dialog.clear()
            with self.dialog:
                ui.image(image)

            self.dialog.open()

        except Exception as err:
            ui.notify(err, type="negative")

    @ui.refreshable
    def ui_upper(self):
        df = self.qc_db.get_last()

        with ui.grid(columns=6):
            for column in df.columns:
                # for column in ["top_weft_edge"]:
                try:
                    tolerance = TOLERANCES[column]
                except KeyError:
                    continue

                res = stats.linregress(df["id"], df[column])
                current = df[column].iloc[-1]
                print(column, df[column].mean(), tolerance, res.slope)

                with ui.card():
                    with ui.row():
                        if float(res.slope) > 0.25:  # TODO fine-tune
                            ui.icon("trending_up", size="4em")
                        elif float(res.slope) < -0.25:
                            ui.icon("trending_down", size="4em")
                        else:
                            ui.icon("trending_flat", size="4em")
                        with ui.column():
                            ui.label(column)
                            if tolerance[0] <= current <= tolerance[1]:
                                ui.label(current).tailwind().text_color("green-400")
                            else:
                                ui.label(current).tailwind().text_color("red-400")

    @ui.refreshable
    def ui_lower(self):
        try:
            df = self.qc_db.retrieve_quality_checks(self.choices[self.selection]["query"])
        except RuntimeError as err:
            ui.notify(err, type="negative")
            return

        table = ui.table.from_pandas(df, title="Datenübersicht", pagination=20)
        table.on('rowClick', self._on_table_row_click)

        label_suffix = ""
        try:
            rule = self.choices[self.selection]["resample"]
            df = df.resample(rule).mean()
            label_suffix = f" (mean @ {rule} intervals)"
        except KeyError:
            pass  # resample is optional

        with ui.grid(columns=2):
            plot("Weft Edge Messungen", df, ["top_weft_edge", "bottom_weft_edge"],
                 label_suffix=label_suffix, tolerances=TOLERANCES["top_weft_edge"])
            plot("Warp Edge Messungen", df, ["left_warp_edge", "right_warp_edge"],
                 label_suffix=label_suffix, tolerances=TOLERANCES["left_warp_edge"])
            plot("Weft Circle", df, ["front_weft_circle", "back_weft_circle"],
                 label_suffix=label_suffix, tolerances=TOLERANCES["front_weft_circle"])
            plot("Lochabstand", df, ["circle_to_circle"], label_suffix=label_suffix,
                 tolerances=TOLERANCES["circle_to_circle"])
            plot("Reconstruction Errors", df, ["reconstruction_error"], label_suffix=label_suffix)
            plot("Materialfehleranzahl", df, ["material_errors"], label_suffix=label_suffix)