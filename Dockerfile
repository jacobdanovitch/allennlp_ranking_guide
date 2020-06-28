FROM allennlp/allennlp:latest

ADD ./requirements.txt /stage/allennlp/requirements.txt
RUN pip install -r requirements.txt

# ADD ./* /stage/allennlp/
ENV PATH "$PATH:/stage/allennlp/scripts"