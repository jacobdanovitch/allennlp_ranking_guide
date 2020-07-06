FROM allennlp/allennlp:latest

ADD ./requirements.txt /stage/allennlp/requirements.txt
RUN pip install -r requirements.txt

ENV PATH "$PATH:/stage/allennlp/scripts"
CMD python scripts/data_split.py https://github.com/microsoft/MIMICS/blob/master/data/MIMICS-ClickExplore.tsv\?raw\=true