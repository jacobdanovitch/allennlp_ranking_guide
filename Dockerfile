FROM allennlp/allennlp:latest

ADD ./requirements.txt /stage/allennlp/requirements.txt
RUN pip install -r requirements.txt

ENV PATH "$PATH:/stage/allennlp/scripts"