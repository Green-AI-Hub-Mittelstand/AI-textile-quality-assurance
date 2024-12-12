#!/home/gaih/koestler/venv/bin/python

import argparse
import asyncio
import random
from datetime import datetime
from multiprocessing import Pipe, Process

import cv2
from nicegui import ui

from data_transfer.dtos import EvalBox
from hmi.hmi_main import HMI
from libs.database import QualityCheckDB
from libs.dummy import get_random_dummy_image
from libs.hardware import send_command
from libs.image_format_conversion import convert_to_opencv
from libs.preprocessing import image_crop
from processes import measurement_process, anomaly_detect_process, material_error_process, scanner_process

RECONSTRUCTION_ERROR_THRESHOLD = 0.01

async def _show_error(msg: str):
    ui.notify(msg, type="negative")
    send_command("error")
    await asyncio.sleep(1)
    send_command("ready")

async def scan_loop():
    while True:
        while not scan_parent_conn.poll():  # wait for results
            await asyncio.sleep(0.1)

        result = scan_parent_conn.recv()
        if result[0]:
            try:
                await process_image(result[1])
            except Exception as ex:
                await _show_error(str(ex))
        else:
            await _show_error(result[1])

        await asyncio.sleep(0.1)


async def process_image(img_pil):
    if not args.dummy:
        send_command("processing")

    processing_done = False

    async def _process_image_ui(notification):
        while not processing_done:
            await asyncio.sleep(0.1)

        notification.message = "Finished processing image!"
        notification.spinner = False
        await asyncio.sleep(1)
        notification.dismiss()

    # handle ui notification as background task
    # timeout required on rare occasions when something goes wrong somewhere...
    asyncio.ensure_future(_process_image_ui(ui.notification(message="Processing image...", spinner=True, timeout=30)))

    before = datetime.now()
    img = convert_to_opencv(img_pil)  # 1 sec
    print(f"Converting to OpenCV took {datetime.now() - before}!")

    before = datetime.now()
    image_cropped = image_crop(img)
    print(f"Cropping took {datetime.now() - before}!")

    before = datetime.now()
    hmi.clear_everything()
    await hmi.update_crop_image(image_cropped)
    print(f"HMI Crop Image Update took {datetime.now() - before}!")


    before = datetime.now()

    # parallel processing using processes
    measure_parent_conn.send(image_cropped)
    #homology_parent_conn.send(image_cropped)
    anomaly_parent_conn.send(image_cropped)
    material_error_parent_conn.send(image_cropped)

    while True:  # wait for results
        if (measure_parent_conn.poll()
                #and homology_parent_conn.poll()
                and anomaly_parent_conn.poll()
                and material_error_parent_conn.poll()):
            break
        await asyncio.sleep(0.1)

    print(f"Parallel processing took {datetime.now() - before}!")

    measure_results = measure_parent_conn.recv()
    # homology_results = homology_parent_conn.recv()
    material_error_results: list[EvalBox] = material_error_parent_conn.recv()
    reconstructed_image, reconstruction_error = anomaly_parent_conn.recv()

    rows = [{"check": "material_errors",
             "result": len(material_error_results) == 0,
             "actual": len(material_error_results),
             "target": 0}]

    if len(material_error_results):
        print("Drawing material errors and updating hmi image...")
        for box in material_error_results:
            print(box.label, box.precision)

            # draw eval boxes into image
            cv2.rectangle(image_cropped, box.top_left, box.bottom_right, color=(0, 255, 0), thickness=10)
            # draw_text(image_cropped, text=f"{box.label} ({box.precision:.2f}", text_position=box.top_left)

        # update HMI
        await hmi.update_crop_image(image_cropped)


    anomaly_result = bool(reconstruction_error <= RECONSTRUCTION_ERROR_THRESHOLD)
    rows.append({'check': 'reconstruction_error',
             'result': anomaly_result,
             'actual': float(reconstruction_error),
             'target': f"<= {RECONSTRUCTION_ERROR_THRESHOLD}"})


    image_measurements = image_cropped.copy()

    qc_result = anomaly_result
    for idx, measurement in enumerate(measure_results):  # type: int, distance_measurement
        variance = measurement.variance
        rows.append({
            'check': measurement.name,
            'result': measurement.is_ok,
            'actual': measurement.distance,
            'target': f'{variance[0]} - {variance[1]}'
        })
        if not measurement.is_ok:
            qc_result = False

        # draw features into image
        p_1 = (int(measurement.p_1[0]), int(measurement.p_1[1]))
        p_2 = (int(measurement.p_2[0]), int(measurement.p_2[1]))
        c = idx % 9
        color = [250 * (c & 1), 250 * (c & 2), 250 * (c & 4)]

        cv2.line(image_measurements, p_1, p_2, color, 20)

    for box in material_error_results:
        # draw eval boxes into image
        cv2.rectangle(image_measurements, box.top_left, box.bottom_right, color=(0, 255, 0), thickness=10)
        qc_result = False


    print("Updating QC result rows...")
    await hmi.update_qc_results(rows)

    print("Updating Measurement image...")
    await hmi.update_measure_image(image_measurements)

    print("Updating Reconstructed image...")
    await hmi.update_reconstructed_image(reconstructed_image)

    print("Saving to database...")
    qc_db.insert_quality_check(qc_result, rows, image_cropped)

    processing_done = True

    if not args.dummy:
        if qc_result:
            send_command("ok")
        else:
            send_command("nok")



async def dummy_scan_loop():
    while True:
        notification = ui.notification(message="Scanning...", spinner=True, timeout=None)
        await asyncio.sleep(5)  # Simulates scanning process
        new_image = get_random_dummy_image()
        notification.dismiss()
        await asyncio.sleep(0.1)

        print("Processing dummy image...")
        await process_image(new_image)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='GAIH Köstler Demonstration')
    parser.add_argument("--dummy", help="use dummy scan images for development without the scanner",
                        action='store_true')
    parser.add_argument("--dpi", type=int, default=600, help="DPI setting of Scanner")
    parser.add_argument("--store-scans", help="Store scans additionally as files", action="store_true")
    args = parser.parse_args()

    # Setup Processes and their connections to main process
    measure_parent_conn, measure_child_conn = Pipe()
    measure_process = Process(target=measurement_process, args=(measure_child_conn, args.dpi), name="Measurement")
    measure_process.start()

    anomaly_parent_conn, anomaly_child_conn = Pipe()
    anomaly_process = Process(target=anomaly_detect_process, args=(anomaly_child_conn,), name="Anomaly Detection")
    anomaly_process.start()

    material_error_parent_conn, material_error_child_conn = Pipe()
    material_error_process = Process(target=material_error_process, args=(material_error_child_conn, args.dpi),
                                     name="Material Error Detection")
    material_error_process.start()

    # homology_parent_conn, homology_child_conn = Pipe()
    # homology_process = Process(target=homology_process, args=(homology_child_conn,), name="Homology")
    # homology_process.start()

    if not args.dummy:
        scan_parent_conn, scan_child_conn = Pipe()
        scan_process = Process(target=scanner_process, args=(scan_child_conn, args.dpi, args.store_scans), name="Scanner")
        scan_process.start()

    qc_db = QualityCheckDB()

    hmi = HMI()

    # start scan loop (dummy if corresponding argument was given)
    ui.timer(0.1, dummy_scan_loop if args.dummy else scan_loop, once=True)

    if not args.dummy:
        send_command("ready")

    ui.run(reload=False, port=6969, title="GAIH Köstler Demonstration HMI")
