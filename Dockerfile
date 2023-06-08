FROM ubuntu:20.04
COPY requirements.txt requirements.txt
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update && apt-get install -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev libgl1 software-properties-common wget && \
    apt-get install -y --no-install-recommends python3-pip python3.8-dev python3.8-distutils python3.8-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    # pip install -r requirements.txt

RUN pip install -r requirements.txt && \
    apt-get update && \
    apt install --no-install-recommends -y poppler-utils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


CMD python3 resume_controller.py