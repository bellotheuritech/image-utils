FROM ubuntu:16.04

ADD debian-requirements.txt .
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
    DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" debian-requirements.txt) && \
    apt-get install -y --force-yes $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm debian-requirements.txt

ADD requirements.txt .
RUN pip3 install -r requirements.txt && \
    rm requirements.txt

WORKDIR /workspace
