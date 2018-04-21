FROM continuumio/anaconda3:latest

RUN /opt/conda/bin/conda update -y -n base conda

RUN /opt/conda/bin/conda install -c conda-forge -y --quiet xgboost
RUN /opt/conda/bin/conda install -y --quiet pymongo

RUN useradd -m myuser
USER myuser

WORKDIR /home/myuser

RUN mkdir sources
RUN mkdir input

COPY ./sources/houseprices sources/houseprices
COPY ./sources/logging.conf sources/logging.conf
COPY ./sources/run.py sources/run.py
COPY ./input/train.csv input/train.csv

WORKDIR /home/myuser/sources

RUN mkdir instance

VOLUME /home/myuser/sources/instance

EXPOSE 5000

CMD /opt/conda/bin/python run.py
#CMD bash