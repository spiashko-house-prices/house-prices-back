FROM continuumio/anaconda3:latest

RUN /opt/conda/bin/conda update -y -n base conda

RUN /opt/conda/bin/conda install -c conda-forge -y --quiet xgboost
RUN /opt/conda/bin/conda install -y --quiet pymongo

RUN useradd -m myuser
USER myuser

WORKDIR /home/myuser

COPY ./houseprices houseprices
COPY ./run.py run.py

WORKDIR /home/myuser

EXPOSE 5000

CMD /opt/conda/bin/python run.py
#CMD bash