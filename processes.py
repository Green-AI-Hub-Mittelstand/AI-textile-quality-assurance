import time
from datetime import datetime

from multiprocessing.connection import Connection

from libs.preprocessing import replace_grey_with_black_hsv


def scanner_process(conn: Connection, dpi: int, store_scans = False):
    from libs.scanner import Scanner
    scanner = Scanner()

    while True:
        try:
            scanned_image = scanner.scan_document(dpi)
            print("Successfully scanned!")
            conn.send((True, scanned_image))
            if store_scans:
                # Erstelle einen eindeutigen Dateinamen mit Zeitstempel
                file_name = f"scan_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
                scanned_image.save(file_name)
                print(f"Bild gespeichert als {file_name}.")

        except Exception as ex:
            if not "feeder out of" in str(ex):
                print(ex)
                conn.send((False, str(ex)))
                time.sleep(1)



def anomaly_detect_process(conn: Connection):
    from Autoencoder.test import AnomalyDetectionAutoencoder

    try:
        anomaly_detector = AnomalyDetectionAutoencoder("Autoencoder/autoencoder_Final.pth")

        while True:
            print("Anomaly Detect Process: Waiting for input image!")
            cropped_img = conn.recv()  # blocks until something is received
            print("Anomaly Detect Process: Received input image!")

            before = datetime.now()
            black_cropped = replace_grey_with_black_hsv(cropped_img)
            print(f"Changing Background to Black took {datetime.now() - before}!")

            before = datetime.now()
            output_image, reconstruction_error = anomaly_detector.reconstruct_image(black_cropped)
            print(f"Anomaly Detection took {datetime.now() - before}!")

            conn.send((output_image, reconstruction_error))
    except FileNotFoundError:
        while True:  # we still need to receive and send data
            print("Warning! Could not load anomaly detection autoencoder! Anomaly detection inactive!")
            conn.recv()  # blocks until something is received
            conn.send((None, 0))

def homology_process(conn: Connection):
    from err_detection.boundary_evaluation import HomologyDetector

    homology_detector = HomologyDetector()

    while True:
        print("Homology Process: Waiting for input image!")
        cropped_img = conn.recv()  # blocks until something is received
        print("Homology Process: Received input image!")
        before = datetime.now()
        homology_results = homology_detector.analyse(cropped_img)
        print(f"Homology took {datetime.now() - before}!")
        conn.send(homology_results)


def measurement_process(conn: Connection, dpi: int):
    from measurement_analysis.measurement_evaluation import MeasurementEvaluator

    measurement_evaluator = MeasurementEvaluator()

    while True:
        print("Measurement Process: Waiting for input image!")
        cropped_img = conn.recv()  # blocks until something is received
        print("Measurement Process: Received input image!")
        before = datetime.now()
        results = measurement_evaluator.analyse(cropped_img, dpi)
        print(f"Measurement took {datetime.now() - before}!")
        conn.send(results)


def material_error_process(conn: Connection, dpi: int):
    from err_detection.material_evaluation import MaterialErrorDetector

    if dpi == 600:
        material_error_detector = MaterialErrorDetector()
    elif dpi == 300:
        material_error_detector = MaterialErrorDetector(size=512)
    else:
        raise RuntimeError("Invalid DPI setting!")

    while True:
        print("Material Error Process: Waiting for input image!")
        cropped_img = conn.recv()  # blocks until something is received
        print("Material Error Process: Received input image!")
        before = datetime.now()
        results = material_error_detector.analyse(cropped_img, 0.8)
        print(f"Material Error Detection took {datetime.now() - before}!")
        conn.send(results)