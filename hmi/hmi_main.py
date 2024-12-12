import asyncio
from typing import Any

import cv2
from PIL import Image
from nicegui import app, ui

from hmi.trend_plots import TrendPlots
from libs.image_format_conversion import convert_opencv_to_base64

qc_table_columns = [
    {'name': 'check', 'label': 'QC Check', 'field': 'check', 'required': True, 'sortable': True},
    {'name': 'result', 'label': 'Result', 'field': 'result', 'required': True, 'sortable': True},
    {'name': 'actual', 'label': 'Ist', 'field': 'actual'},
    {'name': 'target', 'label': 'Soll', 'field': 'target'},
]


def _prepare_image(cv_image: cv2.typing.MatLike) -> str:
    rotated = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return convert_opencv_to_base64(cv2.resize(rotated, None, fx=HMI.SCALE_FACTOR, fy=HMI.SCALE_FACTOR))


class HMI:
    SCALE_FACTOR = 0.25

    def __init__(self):

        self._empty_image = Image.new("RGB", (600, 400), (200,200,200)) # TODO

        app.add_static_files('/static', './hmi/static')

        with ui.splitter(value=10).classes('w-full') as splitter:
            with splitter.before:
                with ui.tabs().props('vertical').classes('w-full') as tabs:
                    # self._tab_setup = ui.tab('setup', label='Einstellungen', icon='settings')
                    self._tab_qc = ui.tab('qc', 'Qualitätskontrolle', icon='rule')
                    self._tab_trend = ui.tab('trend', label='Trendauswertung', icon='assessment')
            with splitter.after:
                with ui.tab_panels(tabs, value=self._tab_qc).props('vertical').classes('w-full h-full'):
                    with ui.tab_panel(self._tab_qc):
                        with ui.splitter().classes('w-full') as qc_splitter:
                            with qc_splitter.before:
                                with ui.tabs().classes('w-full') as image_tabs:  # .style('color: #fff; background-color: #37c346')
                                    self._tab_image_crop = ui.tab('crop', label="Eingabebild", icon="content_cut")
                                    self._tab_image_measure = ui.tab('measure', label="Abmessungen", icon="square_foot")
                                    self._tab_image_reconstructed = ui.tab('reconstructed', label="Anomalie Visu.")

                                with ui.tab_panels(image_tabs, value=self._tab_image_crop).classes('w-full'):
                                    with ui.tab_panel(self._tab_image_crop):
                                        self._image_crop = ui.image(self._empty_image)  # bind_source doesn't work for some reason...
                                    with ui.tab_panel(self._tab_image_measure):
                                        self._image_measure = ui.image(self._empty_image)
                                    with ui.tab_panel(self._tab_image_reconstructed):
                                        self._image_reconstructed = ui.image(self._empty_image)

                            with qc_splitter.after:
                                self._qc_table = ui.table(columns=qc_table_columns, rows=[], row_key='check')
                                self._qc_table.add_slot('body-cell-result', '''
                                    <q-td key="result" :props="props">
                                        <q-badge :color="props.value ? 'green' : 'red'">
                                            {{ props.value ? 'OK' : 'NOK' }}
                                        </q-badge>
                                    </q-td>
                                ''')

                    with ui.tab_panel(self._tab_trend):
                        TrendPlots()

        # with ui.footer().classes('justify-center').style('background-color: #14144b'):
        with ui.footer().classes('justify-end').style('background-color: #37c346'):
            with ui.row():
                ui.image("/static/bmuv_logo_2021.svg").props(f"width=190px height=50px")
                ui.image("/static/dfki_Logo_digital_black.svg").props(f"width=59px height=50px")

    def clear_everything(self):
        self._image_crop.set_source(self._empty_image)
        self._image_measure.set_source(self._empty_image)
        self._image_reconstructed.set_source(self._empty_image)
        self._qc_table.update_rows([])

    async def update_crop_image(self, cv_image: cv2.typing.MatLike):
        try:
            self._image_crop.set_source(_prepare_image(cv_image))
        except cv2.error:
            print("Warning: Could not update crop image!")
            self._image_crop.set_source(self._empty_image)
        await asyncio.sleep(0)

    async def update_measure_image(self, cv_image: cv2.typing.MatLike):
        try:
            self._image_measure.set_source(_prepare_image(cv_image))
        except cv2.error:
            print("Warning: Could not update measure image!")
            self._image_measure.set_source(self._empty_image)
        await asyncio.sleep(0)

    async def update_reconstructed_image(self, cv_image: cv2.typing.MatLike):
        try:
            self._image_reconstructed.set_source(_prepare_image(cv_image))
        except cv2.error:
            print("Warning: Could not update reconstructed image!")
            self._image_reconstructed.set_source(self._empty_image)
        await asyncio.sleep(0)

    async def update_qc_results(self, rows: list[dict[str, Any]]):
        hmi_rows = []
        for row in rows:
            copy = row.copy()
            copy["actual"] = f'{copy["actual"]:.1f}'  # limit decimal parts
            hmi_rows.append(copy)
        self._qc_table.update_rows(hmi_rows)
        await asyncio.sleep(0)



