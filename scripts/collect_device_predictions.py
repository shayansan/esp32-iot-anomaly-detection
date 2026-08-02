#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import serial


HEADER = [
    "timestamp_ms",
    "temperature_c",
    "humidity_percent",
    "light_raw",
    "ground_truth",
    "predicted_anomaly",
    "anomaly_score",
    "inference_time_us",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        required=True,
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def parse_row(line: str):
    fields = line.split(",")

    if len(fields) != 8:
        return None

    try:
        return [
            int(fields[0]),
            float(fields[1]),
            float(fields[2]),
            float(fields[3]),
            int(fields[4]),
            int(fields[5]),
            float(fields[6]),
            int(fields[7]),
        ]
    except ValueError:
        return None


def main() -> None:
    args = parse_arguments()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    collected = 0

    with serial.Serial(
        args.port,
        args.baud,
        timeout=2,
    ) as device:
        print(
            "Press RESET/EN on the ESP32."
        )

        while True:
            line = device.readline().decode(
                "utf-8",
                errors="ignore",
            ).strip()

            if line.startswith("timestamp_ms"):
                break

        with args.output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(HEADER)

            while collected < args.samples:
                line = device.readline().decode(
                    "utf-8",
                    errors="ignore",
                ).strip()

                row = parse_row(line)

                if row is None:
                    continue

                writer.writerow(row)
                collected += 1

                if collected % 100 == 0:
                    output_file.flush()

                    print(
                        f"Collected "
                        f"{collected}/"
                        f"{args.samples}"
                    )

    print(
        f"Saved to {args.output}"
    )


if __name__ == "__main__":
    main()