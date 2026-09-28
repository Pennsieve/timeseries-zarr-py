FROM python:3.12

WORKDIR /app

RUN apt clean && apt-get update && apt-get -y install libhdf5-dev

COPY timeseries_zarr/requirements.txt /app/timeseries_zarr/requirements.txt

RUN pip install -r /app/timeseries_zarr/requirements.txt

COPY timeseries_zarr/ /app/timeseries_zarr

# pynwb/platformdirs needs a writable cache dir even when the container
# runs as a non-root user (HOME may be unset/"/" in the compute env)
ENV XDG_CACHE_HOME=/tmp/.cache

ENV PYTHONPATH="/app"

CMD ["python3.12", "-m", "timeseries_zarr.main"]
