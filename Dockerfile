FROM ecvl

RUN apt-get -y update && apt-get -y install --no-install-recommends \
      python3-dev \
      python3-pip

RUN python3 -m pip install --upgrade --no-cache-dir \
      setuptools pip numpy pybind11 pytest


# Install PyEDDL. Assumes recursive submodule update.
COPY third_party/pyeddl /pyeddl
WORKDIR /pyeddl
RUN python3 setup.py install


# Install PyECVL. Assumes recursive submodule update.
COPY . /pyecvl
WORKDIR /pyecvl
RUN python3 setup.py install
