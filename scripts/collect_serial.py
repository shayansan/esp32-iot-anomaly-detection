#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import serial


CSV_HEADER = [
    "timestamp_ms",
    "temperature_c",
    "humidity_percent",
    "light_raw",
    "anomaly",
]


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Collect labeled sensor data from the ESP32."
    )

    parser.add_argument(
        "--port",
        required=True,
        help="Serial port, for example /dev/ttyACM0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate",
    )

    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="Number of valid CSV rows to collect",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV file",
    )

    return parser.parse_args()


def parse_sensor_row(line: str) -> list[int | float] | None:
    """
    Convert one ESP32 CSV line into numeric values.

    Return None when the line is not a valid sensor row.
    """

    if not line:
        return None

    # Ignore metadata such as "# dataset_seed=42".
    if line.startswith("#"):
        return None

    # Ignore the CSV header.
    if line.startswith("timestamp_ms"):
        return None

    fields = line.split(",")

    if len(fields) != 5:
        return None

    try:
        timestamp_ms = int(fields[0])
        temperature_c = float(fields[1])
        humidity_percent = float(fields[2])
        light_raw = float(fields[3])
        anomaly = int(fields[4])
    except ValueError:
        return None

    # Reject invalid anomaly labels.
    if anomaly not in (0, 1):
        return None

    return [
        timestamp_ms,
        temperature_c,
        humidity_percent,
        light_raw,
        anomaly,
    ]


def wait_for_header(device: serial.Serial) -> None:
    """
    Wait until the ESP32 prints its CSV header.

    If the ESP32 is already running, press its RESET/EN button.
    """

    print("Waiting for the ESP32 CSV header...")
    print("Press the ESP32 RESET/EN button if necessary.")

    while True:
        raw_line = device.readline()

        if not raw_line:
            continue

        line = raw_line.decode(
            "utf-8",
            errors="ignore",
        ).strip()

        print(f"ESP32: {line}")

        if line.startswith("timestamp_ms"):
            print("CSV header detected.")
            return


def main() -> None:
    args = parse_arguments()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    collected_samples = 0

    with serial.Serial(
        port=args.port,
        baudrate=args.baud,
        timeout=2.0,
    ) as device:
        wait_for_header(device)

        with args.output.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(CSV_HEADER)

            while collected_samples < args.samples:
                raw_line = device.readline()

                if not raw_line:
                    continue

                line = raw_line.decode(
                    "utf-8",
                    errors="ignore",
                ).strip()

                row = parse_sensor_row(line)

                if row is None:
                    print(f"Ignored line: {line}")
                    continue

                writer.writerow(row)
                collected_samples += 1

                if collected_samples % 100 == 0:
                    output_file.flush()

                    print(
                        f"Collected "
                        f"{collected_samples}/"
                        f"{args.samples} samples"
                    )

    print(
        f"Dataset saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()